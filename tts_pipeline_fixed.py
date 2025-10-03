#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tts_pipeline_fixed.py

完整修正版：
 - CosyVoice 多签名容错调用
 - 临时文件保持 .tmp.wav，避免 torchaudio/ffmpeg 因 tmp 后缀失败
 - WAV->OGG 转码多方法尝试（pydub(libopus/libvorbis) -> ffmpeg subprocess(libopus/libvorbis)）
 - 失败时保留 WAV 并写日志，不生成 0KB 的 OGG
 - 每次 queue.get() 后确保 task_done()
 - sentinel (None) 优雅退出线程
 - 生成成功后写 .done 并记录到 SQLite processed 表
"""

import os
import sys
import time
import re
import json
import ast
import sqlite3
import traceback
import logging
import subprocess
from queue import Queue
import threading
from tqdm import tqdm
from pathlib import Path

# audio libraries
from pydub import AudioSegment
import torchaudio
from torchaudio.transforms import Resample

# model path configs (按需修改)
MATCHA_TTS_PATH = "/text2audio/KDCCDM_Script/CosyVoice/third_party/Matcha-TTS"
COSYVOICE_PATH = "/text2audio/KDCCDM_Script/CosyVoice"
COSYVOICE_MODEL_PATH = "/text2audio/KDCCDM_Script/CosyVoice/pretrained_models/CosyVoice2-0.5B"
SENSE_VOICE_SMALL_PATH = "/text2audio/KDCCDM_Script/CosyVoice/iic/SenseVoiceSmall"

# I/O config (按需修改)
OGG_INPUT_BASE_PATH = "/text2audio/english-part3"
OGG_INPUT_SEARCH_PATH = "/text2audio/english-part3/dialog"
OGG_OUTPUT_PATH = "/text2audio/voiceout-part3"

# DB / xml
DIALOG_DB_PATH = 'dialog.db'
DIALOG_DB_TABLE = 'dialog'
PROCESSED_TABLE = 'processed'
DIALOG_XML_PATH = 'text_ui_dialog.xml'

# runtime config
BATCH_SIZE = 3
THIS_PART = 1
TOTAL_PART = 1

NUM_SENSE_WORKERS = 2
NUM_TRANSLATE_WORKERS = 1
NUM_COZY_WORKERS = 1

# logging
logging.basicConfig(
    filename="tts_pipeline.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logging.getLogger().addHandler(console)

# ensure paths exist or warn
for p, name in [
    (MATCHA_TTS_PATH, "MATCHA_TTS_PATH"),
    (COSYVOICE_PATH, "COSYVOICE_PATH"),
    (COSYVOICE_MODEL_PATH, "COSYVOICE_MODEL_PATH"),
    (SENSE_VOICE_SMALL_PATH, "SENSE_VOICE_SMALL_PATH"),
    (OGG_INPUT_BASE_PATH, "OGG_INPUT_BASE_PATH"),
    (OGG_INPUT_SEARCH_PATH, "OGG_INPUT_SEARCH_PATH"),
]:
    if not os.path.exists(p):
        logging.warning(f"配置 {name} = {p} 不存在，请确认是否需要修改。")

os.makedirs(OGG_OUTPUT_PATH, exist_ok=True)

# ---------------------------
# DB / XML -> dialog db init
# ---------------------------
import xml.etree.ElementTree as ET

class DialogMatcher:
    def __init__(self, db_path=DIALOG_DB_PATH):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.c = self.conn.cursor()
        self.init_db()
        self.conn.create_function("REGEXP", 2, DialogMatcher.regexp)

    def init_db(self):
        self.c.execute(f'CREATE TABLE IF NOT EXISTS {DIALOG_DB_TABLE} (id text primary key, eng_text text, chn_text text)')
        self.c.execute(f'CREATE TABLE IF NOT EXISTS {PROCESSED_TABLE} (id TEXT PRIMARY KEY, src TEXT, out TEXT, ts DATETIME)')
        self.conn.commit()

    def load_from_xml(self, dialog_path):
        if not os.path.exists(dialog_path):
            logging.warning(f"dialog xml 未找到: {dialog_path}")
            return
        tree = ET.parse(dialog_path)
        root = tree.getroot()
        count = 0
        for row in root:
            rowid = row[0].text
            eng_text = row[1].text
            chn_text = row[2].text
            self.c.execute(f"INSERT OR REPLACE INTO {DIALOG_DB_TABLE} VALUES (?, ?, ?)", (rowid, eng_text, chn_text))
            count += 1
        self.conn.commit()
        logging.info(f"Loaded {count} rows into dialog db from xml")

    @staticmethod
    def regexp(expr, item):
        import re as _re
        reg = _re.compile(expr)
        return reg.search(item) is not None

    def match_file(self, file_name):
        """
        线程安全的 match_file：每次新建短连接执行查询，避免复用同一 cursor 导致并发错误。
        返回与原实现相同的 row 或 None。
        """
        # 使用新的本地连接，确保线程间不会复用同一 cursor
        conn = sqlite3.connect(DIALOG_DB_PATH, check_same_thread=True)
        try:
            # 重新注册 REGEXP 函数到这个连接
            def _regexp_local(expr, item):
                return re.compile(expr).search(item) is not None
            conn.create_function("REGEXP", 2, _regexp_local)
            cur = conn.cursor()

            id = os.path.splitext(os.path.basename(file_name))[0]
            parts = id.split('_')
            for i in range(len(parts)):
                joined_id = '_'.join(parts[i:])
                cur.execute(f"SELECT * FROM {DIALOG_DB_TABLE} WHERE id like ? and eng_text <> chn_text and eng_text REGEXP '[A-Za-z]'", ('%'+joined_id,))
                row = cur.fetchone()
                if row is not None:
                    # 如果英文本等于中文文本，视为无需处理
                    if row[1] == row[2]:
                        return None
                    return row
            return None
        finally:
            try:
                cur.close()
            except:
                pass
            conn.close()

    def mark_processed(self, id, src, out):
        """
        线程安全地将处理结果写入 processed 表：每次使用独立短连接写入。
        """
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        # 每次单独建立连接以避免和其他线程复用 cursor
        conn = sqlite3.connect(DIALOG_DB_PATH, check_same_thread=True)
        try:
            c = conn.cursor()
            c.execute(f"INSERT OR REPLACE INTO {PROCESSED_TABLE} VALUES (?, ?, ?, ?)", (id, src, out, ts))
            conn.commit()
        except Exception:
            logging.exception("写 processed 表失败")
        finally:
            try:
                c.close()
            except:
                pass
            conn.close()


matcher = DialogMatcher(DIALOG_DB_PATH)
matcher.load_from_xml(DIALOG_XML_PATH)

# ---------------------------
# Reader
# ---------------------------
def get_partition(lst, cur_part, total_part):
    part_size = len(lst) // total_part
    remainder = len(lst) % total_part
    start_idx = (cur_part - 1) * part_size + min(cur_part - 1, remainder)
    end_idx = start_idx + part_size + (1 if cur_part <= remainder else 0)
    return lst[start_idx:end_idx]

class Reader:
    def __init__(self, root_path):
        self.root_path = os.path.abspath(root_path)
        self.files = []
        for dirpath, dirnames, filenames in os.walk(self.root_path):
            for filename in filenames:
                abs_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(abs_path, self.root_path)
                self.files.append({
                    "absolute_path": abs_path,
                    "relative_path": rel_path,
                    "filename": filename
                })
        self.files.sort(key=lambda x: x["absolute_path"])
        self.files = get_partition(self.files, THIS_PART, TOTAL_PART)
        logging.info(f"Reader read {len(self.files)} files from {self.root_path}")

# ---------------------------
# AudioConverter helper
# ---------------------------
class AudioConverter:
    def __init__(self):
        pass

    def replace_extension(self, input_path, new_extension):
        base, _ = os.path.splitext(input_path)
        return f"{base}{new_extension}"

    def ogg2wav(self, input_file, output_file=None):
        if not output_file:
            output_file = os.path.splitext(input_file)[0] + ".wav"
        if os.path.exists(output_file):
            return output_file
        outdir = os.path.dirname(output_file)
        if outdir and not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)
        audio = AudioSegment.from_file(input_file)
        audio = audio.set_frame_rate(16000)
        audio.export(output_file, format="wav")
        return output_file

# ---------------------------
# SenseVoiceInference
# ---------------------------
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

class SenseVoiceInference:
    def __init__(self, model_dir, device="cuda:0", ncpu=4, output_dir=None, batch_size=1, hub="ms", **kwargs):
        self.model_dir = model_dir
        self.device = device
        self.ncpu = ncpu
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.hub = hub
        self.kwargs = kwargs
        logging.info("Loading SenseVoice model...")
        self.model = AutoModel(
            model=model_dir,
            device=device,
            ncpu=ncpu,
            output_dir=output_dir,
            batch_size=batch_size,
            hub=hub,
            disable_log=True,
            disable_update=True,
            **kwargs
        )
        logging.info("SenseVoice model loaded")

    def call(self, ogg_file_list, language="en", use_itn=True, **kwargs):
        self.results = []
        total_files = len(ogg_file_list)
        for start in range(0, total_files, self.batch_size):
            batch_files = ogg_file_list[start:start + self.batch_size]
            batch_files = [f for f in batch_files if os.path.exists(f)]
            if not batch_files:
                continue
            logging.info(f"SenseVoice processing batch starting {start}, size {len(batch_files)}")
            t0 = time.time()
            batch_res = self.model.generate(
                input=batch_files,
                language=language,
                use_itn=use_itn,
                **kwargs
            )
            logging.info(f"SenseVoice batch took {time.time()-t0:.2f}s")
            for i, item in enumerate(batch_res):
                text = rich_transcription_postprocess(item.get("text",""))
                self.results.append({"file_path": batch_files[i], "text": text})
        return self.results

# ---------------------------
# CosyVoice integration (robust)
# ---------------------------
sys.path.insert(0, os.path.abspath(MATCHA_TTS_PATH))
sys.path.insert(0, os.path.abspath(COSYVOICE_PATH))
try:
    from CosyVoice.cosyvoice.cli.cosyvoice import CosyVoice2
except Exception:
    logging.exception("无法导入 CosyVoice 模块，请确认 COSYVOICE_PATH 配置正确和模块可用")
    raise

class CozyVoiceInference:
    def __init__(self, device=None):
        self.cosyvoice = self._load_model(device)
        self.sample_rate = getattr(self.cosyvoice, "sample_rate", 16000)

    def _load_model(self, device):
        model_path = os.path.abspath(COSYVOICE_MODEL_PATH)
        model = CosyVoice2(
            model_path,
            load_jit=True,
            load_onnx=False,
            load_trt=False,
        )
        return model

    def call(self, text, prompt, prompt_speech_path, output_path, stream=False):
        """
        text: 中文文本
        prompt: 原始英文文本 (prompt)
        prompt_speech_path: 原始音频路径 (ogg/wav)
        output_path: 目标 wav 路径（最终先写 wav，再转 ogg）
        返回最终写入的 wav 路径（成功）或抛异常
        """
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # load prompt audio (try torchaudio, else pydub fallback)
        try:
            prompt_speech, original_sample_rate = torchaudio.load(prompt_speech_path)
        except Exception:
            tmp_wav = output_path + ".prompt_tmp.wav"
            try:
                AudioSegment.from_file(prompt_speech_path).export(tmp_wav, format="wav")
                prompt_speech, original_sample_rate = torchaudio.load(tmp_wav)
                os.remove(tmp_wav)
            except Exception:
                logging.exception(f"无法加载提示语音: {prompt_speech_path}")
                raise

        if original_sample_rate != 16000:
            resampler = Resample(orig_freq=original_sample_rate, new_freq=16000)
            prompt_speech = resampler(prompt_speech)
            logging.info("Resample prompt cost")

        tmp_out = output_path + ".tmp.wav"  # keep .wav extension
        final_out = output_path

        # robust call to inference_zero_shot with multiple signature attempts
        gen = None
        last_exc = None
        try:
            gen = self.cosyvoice.inference_zero_shot(text, prompt, prompt_speech, stream=stream, speed=1.18)
        except TypeError as e1:
            last_exc = e1
            try:
                gen = self.cosyvoice.inference_zero_shot(prompt_text=text, prompt_speech_16k=prompt_speech, prompt=prompt, stream=stream, speed=1.18)
            except TypeError as e2:
                last_exc = e2
                try:
                    gen = self.cosyvoice.inference_zero_shot(None, text, prompt_speech, stream=stream, speed=1.18)
                except Exception as e3:
                    last_exc = e3
                    try:
                        gen = self.cosyvoice.inference_zero_shot(text, prompt_speech, stream=stream, speed=1.18)
                    except Exception as e4:
                        last_exc = e4
                        logging.exception("CozyVoice inference call failed for all attempted signatures")
                        raise last_exc

        if gen is None:
            raise RuntimeError("Unable to obtain generator/result from cosy.inference_zero_shot")

        # save results to tmp_out then atomic replace
        try:
            if isinstance(gen, dict):
                result = gen
                torchaudio.save(tmp_out, result['tts_speech'], self.sample_rate)
            else:
                for j, result in enumerate(gen):
                    torchaudio.save(tmp_out, result['tts_speech'], self.sample_rate)
            os.replace(tmp_out, final_out)
            try:
                with open(final_out + ".done", "w") as f:
                    f.write(f"generated_from:{prompt_speech_path}\n")
            except Exception:
                logging.warning("写 wav .done 失败")
            return final_out
        except Exception:
            if os.path.exists(tmp_out):
                try:
                    os.remove(tmp_out)
                except:
                    pass
            logging.exception("CozyVoice inference failed during save/replace")
            raise

# ---------------------------
# LLM Translator (no plaintext key)
# ---------------------------
try:
    from openai import OpenAI
except Exception:
    OpenAI = None
    logging.warning("openai.OpenAI not available; ensure openai package is installed if using OpenAI APIs")

class LLMTranslator:
    def __init__(self, max_retries=3):
        self.max_retries = max_retries
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPBRICKS_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("DEEPBRICKS_BASE_URL")
        if not api_key:
            logging.error("未检测到 OPENAI_API_KEY 或 DEEPBRICKS_API_KEY，请设置环境变量。")
            raise RuntimeError("Missing API key")
        if OpenAI is not None:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = None
            logging.warning("OpenAI client not available; translation will fail if used.")

    def call(self, text, model="gpt-4-turbo"):
        try:
            completion = self.client.chat.completions.create(
                model=model,
                messages=[{"role":"user","content":text}]
            )
            response = completion.choices[0].message.content
            return response
        except Exception:
            logging.exception("LLM 调用异常")
            return f"Error: LLM call failed"

    def checklist(self, response):
        try:
            parsed = json.loads(response)
            if isinstance(parsed, list):
                return parsed
        except:
            pass
        try:
            parsed = ast.literal_eval(response)
            if isinstance(parsed, list):
                return parsed
        except:
            pass
        return None

    def __call__(self, sentences, model="gpt-4o-mini"):
        MAX_WORDS_PER_CALL = 680
        total_words = sum(len(s.split()) for s in sentences)
        if total_words <= MAX_WORDS_PER_CALL:
            return self.process_sentences(sentences, model)
        batch_size = MAX_WORDS_PER_CALL // max(1, max(len(s.split()) for s in sentences))
        results = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            br = self.process_sentences(batch, model)
            if br is None:
                return None
            results.extend(br)
        return results

    def process_sentences(self, sentences, model):
        prompt = f"translate following sentences into chinese, return just a list structure like: ['s1','s2'], without other words\n{str(sentences)}"
        retries = 0
        while retries <= self.max_retries:
            response = self.call(prompt, model=model)
            parsed = self.checklist(response)
            if parsed is not None:
                return parsed
            retries += 1
            time.sleep(1)
        logging.error("LLM 翻译多次重试失败")
        return None

# ---------------------------
# WAV -> OGG robust converter
# ---------------------------
def convert_wav_to_ogg(out_wav, out_ogg):
    """
    Try multiple methods to convert WAV -> OGG robustly.
    Returns True if conversion succeeded and out_ogg exists & non-empty, else False.
    """
    # remove any existing broken target
    try:
        if os.path.exists(out_ogg):
            os.remove(out_ogg)
    except:
        pass

    # 1) Try pydub with libopus
    try:
        audio_seg = AudioSegment.from_file(out_wav, format="wav")
        audio_seg.export(out_ogg, format="ogg", codec="libopus", bitrate="128k")
        if os.path.exists(out_ogg) and os.path.getsize(out_ogg) > 100:
            return True
    except Exception as e:
        logging.debug("pydub(libopus) failed: %s", repr(e))

    # 2) Try pydub with libvorbis
    try:
        audio_seg = AudioSegment.from_file(out_wav, format="wav")
        audio_seg.export(out_ogg, format="ogg", codec="libvorbis", bitrate="128k")
        if os.path.exists(out_ogg) and os.path.getsize(out_ogg) > 100:
            return True
    except Exception as e:
        logging.debug("pydub(libvorbis) failed: %s", repr(e))

    # 3) Try ffmpeg subprocess with libopus
    try:
        cmd = ["ffmpeg", "-y", "-i", out_wav, "-c:a", "libopus", "-b:a", "128k", out_ogg]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode == 0 and os.path.exists(out_ogg) and os.path.getsize(out_ogg) > 100:
            return True
        logging.debug("ffmpeg(libopus) rc=%s stderr=%s", proc.returncode, proc.stderr[-1000:])
    except Exception as e:
        logging.debug("ffmpeg(libopus) call failed: %s", repr(e))

    # 4) Try ffmpeg subprocess with libvorbis
    try:
        cmd = ["ffmpeg", "-y", "-i", out_wav, "-c:a", "libvorbis", "-q:a", "5", out_ogg]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode == 0 and os.path.exists(out_ogg) and os.path.getsize(out_ogg) > 100:
            return True
        logging.debug("ffmpeg(libvorbis) rc=%s stderr=%s", proc.returncode, proc.stderr[-1000:])
    except Exception as e:
        logging.debug("ffmpeg(libvorbis) call failed: %s", repr(e))

    return False

# ---------------------------
# Utils
# ---------------------------
def is_already_processed(output_path):
    done_path = output_path + ".done"
    if os.path.exists(done_path):
        return True
    exts = [".wav", ".ogg", ".flac", ".mp3", ""]
    for e in exts:
        p = output_path if e == "" else os.path.splitext(output_path)[0] + e
        if os.path.exists(p):
            try:
                if os.path.getsize(p) > 1024:
                    return True
            except:
                return True
    return False

def is_single_word(s):
    s = (s or "").strip()
    if not s:
        return False
    words = re.findall(r'\b\w+\b', s)
    return len(words) == 1

# ---------------------------
# Queues & Workers
# ---------------------------
fileReader = Reader(OGG_INPUT_SEARCH_PATH)
ogg_list = [item["absolute_path"] for item in fileReader.files]
logging.info(f"总文件数量: {len(ogg_list)}")

sensevoice_task_queue = Queue()
translate_task_queue = Queue()
cozyvoice_task_queue = Queue()

audio_converter = AudioConverter()
cozy = CozyVoiceInference(device="cuda:0")
sensevoice_model = SenseVoiceInference(model_dir=SENSE_VOICE_SMALL_PATH, device="cuda:0", batch_size=1)
translate = LLMTranslator()

finish_file_count = 0
finish_file_lock = threading.Lock()

def sensevoice_worker():
    logging.info("SenseVoice worker started")
    while True:
        batch = sensevoice_task_queue.get()
        try:
            if batch is None:
                logging.info("SenseVoice worker got sentinel, exiting")
                break
            try:
                results = sensevoice_model.call(ogg_file_list=batch, language="en")
            except Exception:
                logging.exception("SenseVoice inference failed for batch")
                results = []
            for idx, item in enumerate(results):
                try:
                    translate_task_queue.put((item["text"], batch[idx], item))
                except Exception:
                    logging.exception("put translate task failed")
        except Exception:
            logging.exception("SenseVoice worker exception")
        finally:
            sensevoice_task_queue.task_done()

def translate_worker():
    logging.info("Translate worker started")
    while True:
        task = translate_task_queue.get()
        try:
            if task is None:
                logging.info("Translate worker got sentinel, exiting")
                break
            original_text, ogg_file, result = task
            translated_list = translate([original_text])
            if translated_list is None:
                logging.error(f"Translation failed for {ogg_file}, skip")
                continue
            translated_text = translated_list[0]
            cozyvoice_task_queue.put((translated_text, original_text, ogg_file))
        except Exception:
            logging.exception("Translate worker exception")
        finally:
            translate_task_queue.task_done()

def cozy_worker():
    global finish_file_count
    logging.info("Cozy worker started")
    while True:
        task = cozyvoice_task_queue.get()
        try:
            if task is None:
                logging.info("Cozy worker got sentinel, exiting")
                break
            translated_text, original_text, ogg_file = task
            rel_res_path = ogg_file.replace(OGG_INPUT_BASE_PATH, "").lstrip(os.path.sep)
            res_output_wav = os.path.normpath(os.path.join(OGG_OUTPUT_PATH, rel_res_path))
            res_output_wav = os.path.splitext(res_output_wav)[0] + ".wav"

            if not (re.search(r'[\u4e00-\u9fff]', original_text) or re.search(r'[A-Za-z]', original_text)):
                logging.info(f"Skip {ogg_file} because original text lacks alnum/chinese")
                continue

            # if is_single_word(original_text) and (not translated_text or len(translated_text.strip()) < 4):
            #     logging.info(f"Skip short single-word {original_text} for {ogg_file}")
            #     continue

            # skip if already processed (.ogg)
            if is_already_processed(os.path.splitext(res_output_wav)[0] + ".ogg"):
                logging.info(f"跳过文件 {ogg_file}（已处理）")
                continue

            try:
                out_wav = cozy.call(translated_text, original_text, ogg_file, res_output_wav, stream=False)
            except Exception:
                logging.exception(f"Cozy failed for {ogg_file}")
                continue

            final_out = out_wav
            if out_wav and os.path.exists(out_wav):
                out_ogg = os.path.splitext(out_wav)[0] + ".ogg"
                try:
                    success = convert_wav_to_ogg(out_wav, out_ogg)
                    if success:
                        # remove intermediate wav & its done
                        try:
                            if os.path.exists(out_wav):
                                os.remove(out_wav)
                            if os.path.exists(out_wav + ".done"):
                                os.remove(out_wav + ".done")
                        except Exception:
                            logging.warning("无法删除中间 wav 或 .done")
                        # write done for ogg
                        try:
                            with open(out_ogg + ".done", "w") as f:
                                f.write(f"generated_from:{ogg_file}\n")
                        except Exception:
                            logging.warning("写 ogg .done 失败")
                        final_out = out_ogg
                    else:
                        logging.error(f"WAV->OGG 转码失败 for {out_wav}, keep wav")
                        final_out = out_wav
                except Exception:
                    logging.exception(f"WAV->OGG 转码异常 for {out_wav}, keep wav")
                    final_out = out_wav
            else:
                final_out = out_wav

            # record processed to DB
            file_id = os.path.splitext(os.path.basename(ogg_file))[0]
            matcher.mark_processed(file_id, ogg_file, final_out)
            with finish_file_lock:
                finish_file_count += 1
            logging.info(f"Cozy generated {final_out} for {ogg_file} (total finished {finish_file_count})")

        except Exception:
            logging.exception("Cozy worker exception")
        finally:
            cozyvoice_task_queue.task_done()

# start threads
sense_threads = []
for _ in range(NUM_SENSE_WORKERS):
    t = threading.Thread(target=sensevoice_worker, daemon=True)
    t.start()
    sense_threads.append(t)

translate_threads = []
for _ in range(NUM_TRANSLATE_WORKERS):
    t = threading.Thread(target=translate_worker, daemon=True)
    t.start()
    translate_threads.append(t)

cozy_threads = []
for _ in range(NUM_COZY_WORKERS):
    t = threading.Thread(target=cozy_worker, daemon=True)
    t.start()
    cozy_threads.append(t)

# push tasks
for i in tqdm(range(0, len(ogg_list), BATCH_SIZE), desc="Processing batches"):
    ogg_batch = ogg_list[i:i + BATCH_SIZE]
    filtered = []
    for ogg_file in ogg_batch:
        rel_res_path = ogg_file.replace(OGG_INPUT_BASE_PATH, "").lstrip(os.path.sep)
        res_output_wav = os.path.normpath(os.path.join(OGG_OUTPUT_PATH, rel_res_path))
        res_output_wav = os.path.splitext(res_output_wav)[0] + ".wav"

        if is_already_processed(os.path.splitext(res_output_wav)[0] + ".ogg"):
            logging.info(f"跳过文件 {ogg_file}（已处理）")
            continue

        dialog_from_db = matcher.match_file(ogg_file)
        if dialog_from_db is not None:
            cozyvoice_task_queue.put((dialog_from_db[2], dialog_from_db[1], ogg_file))
        else:
            sensevoice_task_queue.put([ogg_file])

logging.info("All input pushed. Waiting for sensevoice tasks to finish...")
sensevoice_task_queue.join()
logging.info("Sensevoice tasks finished. Sending sentinel to translate queue")

for _ in range(NUM_TRANSLATE_WORKERS):
    translate_task_queue.put(None)

translate_task_queue.join()
logging.info("Translate tasks finished. Sending sentinel to cozy queue")

for _ in range(NUM_COZY_WORKERS):
    cozyvoice_task_queue.put(None)

cozyvoice_task_queue.join()
logging.info("Cozy tasks finished. Waiting threads to join")

for t in sense_threads:
    t.join(timeout=1)
for t in translate_threads:
    t.join(timeout=1)
for t in cozy_threads:
    t.join(timeout=1)

logging.info("All done. Total finished files: {}".format(finish_file_count))

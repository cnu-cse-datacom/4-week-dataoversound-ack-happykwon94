from __future__ import print_function

#201502012 권 오 범

import sys
import wave

from io import StringIO

import alsaaudio
import colorama
import numpy as np
import pyaudio

from reedsolo import RSCodec, ReedSolomonError
from termcolor import cprint
from pyfiglet import figlet_format

HANDSHAKE_START_HZ = 4096
HANDSHAKE_END_HZ = 5120 + 1024

START_HZ = 1024
STEP_HZ = 256
BITS = 4

FEC_BYTES = 4

def stereo_to_mono(input_file, output_file):
    inp = wave.open(input_file, 'r')
    params = list(inp.getparams())
    params[0] = 1 # nchannels
    params[3] = 0 # nframes

    out = wave.open(output_file, 'w')
    out.setparams(tuple(params))

    frame_rate = inp.getframerate()
    frames = inp.readframes(inp.getnframes())
    data = np.fromstring(frames, dtype=np.int16)
    left = data[0::2]
    out.writeframes(left.tostring())

    inp.close()
    out.close()

def yield_chunks(input_file, interval):
    wav = wave.open(input_file)
    frame_rate = wav.getframerate()

    chunk_size = int(round(frame_rate * interval))
    total_size = wav.getnframes()

    while True:
        chunk = wav.readframes(chunk_size)
        if len(chunk) == 0:
            return

        yield frame_rate, np.fromstring(chunk, dtype=np.int16)

def dominant(frame_rate, chunk):
    w = np.fft.fft(chunk)
    freqs = np.fft.fftfreq(len(chunk))
    peak_coeff = np.argmax(np.abs(w))
    peak_freq = freqs[peak_coeff]
    return abs(peak_freq * frame_rate) # in Hz

def match(freq1, freq2):
    return abs(freq1 - freq2) < 20

def decode_bitchunks(chunk_bits, chunks):
    out_bytes = []

    next_read_chunk = 0
    next_read_bit = 0

    byte = 0
    bits_left = 8
    while next_read_chunk < len(chunks):
        can_fill = chunk_bits - next_read_bit
        to_fill = min(bits_left, can_fill)
        offset = chunk_bits - next_read_bit - to_fill
        byte <<= to_fill
        shifted = chunks[next_read_chunk] & (((1 << to_fill) - 1) << offset)
        byte |= shifted >> offset;
        bits_left -= to_fill
        next_read_bit += to_fill
        if bits_left <= 0:

            out_bytes.append(byte)
            byte = 0
            bits_left = 8

        if next_read_bit >= chunk_bits:
            next_read_chunk += 1
            next_read_bit -= chunk_bits

    return out_bytes

def decode_file(input_file, speed):
    wav = wave.open(input_file)
    if wav.getnchannels() == 2:
        mono = StringIO()
        stereo_to_mono(input_file, mono)

        mono.seek(0)
        input_file = mono
    wav.close()

    offset = 0
    for frame_rate, chunk in yield_chunks(input_file, speed / 2):
        dom = dominant(frame_rate, chunk)
        print("{} => {}".format(offset, dom))
        offset += 1

def extract_packet(freqs):
    freqs = freqs[::2]
    bit_chunks = [int(round((f - START_HZ) / STEP_HZ)) for f in freqs]
    bit_chunks = [c for c in bit_chunks[1:] if 0 <= c < (2 ** BITS)]
    return bytearray(decode_bitchunks(BITS, bit_chunks))

def display(s):
    cprint(figlet_format(s.replace(' ', '   '), font='doom'), 'yellow')

def listen_linux(frame_rate=44100, interval=0.1):

    mic = alsaaudio.PCM(alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NORMAL, device="default")
    mic.setchannels(1)
    mic.setrate(44100)
    mic.setformat(alsaaudio.PCM_FORMAT_S16_LE)

    num_frames = int(round((interval / 2) * frame_rate))
    mic.setperiodsize(num_frames)
    print("start...")

    in_packet = False
    packet = []

    while True:
        l, data = mic.read()
        if not l:
            continue

        chunk = np.fromstring(data, dtype=np.int16)
        dom = dominant(frame_rate, chunk)

        if in_packet and match(dom, HANDSHAKE_END_HZ):
            byte_stream = extract_packet(packet)
            try:
                byte_stream = RSCodec(FEC_BYTES).decode(byte_stream)
                byte_stream = byte_stream.decode("utf-8")                         
                #display(byte_stream)

                print("---------------------------------------------")
                
                #초기에 상태 설정
                state = False
                
                #입력 받은 데이터 내부에 학번 정보 검사
                if '201502012' in byte_stream:
                    state = True
                
                #데이터에 학번 정보 있으면 실행
                if state:
                    #학번에 대한 데이터를 제거하고 data만 남긴다.
                    byte_stream_data = byte_stream.replace('201502012', '')
                    print("데이터 추출",byte_stream_data)
                    
                    #다시 인코드 시킨다.
                    byte_stream_data = byte_stream_data.encode("utf-8")
                    
                    #앞뒤의 공백을 제거
                    byte_stream_data = byte_stream_data.strip()
                    print("encode : ",byte_stream_data)
                    
                    #빈 배열 생성
                    data = []
                    for i in byte_stream_data:
                        #1바이트를 4비트씩 분해해서 data 배열에 Hz로 계산하여 삽입
                        front = i >> 4
                        back = i & 0x0f
                        front = front * STEP_HZ + START_HZ
                        back = back * STEP_HZ + START_HZ
                        data.append(front)
                        data.append(back)
                
                    #Hz 정보 출력
                    print("data_Hz : ", data)
                    
                    #pyaudio 객체 생성
                    p = pyaudio.PyAudio()
                    stream = p.open(format=pyaudio.paFloat32, 
                            channels=1, 
                            rate=44100, 
                            output = True)
                    
                    #Hz를 따라서 소리 출력
                    for freq in data:
                        #Hz를 정현파로 만든다.
                        examples = (np.sin(2*np.pi*np.arange(44100*1.1)
                            *freq/44100)).astype(np.float32)
                        print("freq", freq)
                        #소리를 출력한다.
                        stream.write(examples)
                    
                    #초기에 입력받은 값 중에 학번을 제거한 데이터만 출력
                    byte_stream_data = byte_stream_data.decode("utf-8")
                    display(byte_stream_data)

                else:
                    #학번이 일치하지 않으면 오류 문구 출력    
                    print("Match Error")
                         
            except ReedSolomonError as e:
                pass
                #print("{}: {}".format(e, byte_stream))

            packet = []
            in_packet = False
        elif in_packet:
            packet.append(dom)
        elif match(dom, HANDSHAKE_START_HZ):
            in_packet = True
    
if __name__ == '__main__':
    colorama.init(strip=not sys.stdout.isatty())

    #decode_file(sys.argv[1], float(sys.argv[2]))
    listen_linux()

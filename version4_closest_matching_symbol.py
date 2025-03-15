import os
import wave
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import heapq
import time
import bitarray
import sounddevice as sd
import scipy.io.wavfile as wavfile
import sys
import random
# read audio files in the folder
current_dir = os.getcwd()
audio_dir = os.path.join(current_dir, "audio")


wav_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]

audio_samples = []
for wav_file in wav_files:
    wav_path = os.path.join(audio_dir, wav_file)
    with wave.open(wav_path, "rb") as wav_file:
        num_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_rate = wav_file.getframerate()
        num_frames = wav_file.getnframes()
        audio_data = wav_file.readframes(num_frames)

    samples = np.frombuffer(audio_data, dtype=np.int16)

    audio_samples.extend(samples)

audio_samples = np.array(audio_samples)
sample_rate = frame_rate  

total_len = len(audio_samples)
half_len = total_len // 2# 50% as training data
all_indices = np.arange(total_len)
random_indices = np.random.choice(all_indices, size=half_len, replace=False)
training_data = audio_samples[random_indices]


symbol_counts = Counter(training_data)
# Huffman functions
class HuffmanNode:
    def __init__(self, symbol, frequency):
        self.symbol = symbol
        self.frequency = frequency
        self.left = None
        self.right = None
    
    def __lt__(self, other):
        return self.frequency < other.frequency

def build_huffman_tree(symbol_counts):
    heap = [HuffmanNode(symbol, freq) for symbol, freq in symbol_counts.items()]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(None, left.frequency + right.frequency)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)
    
    return heap[0]

def generate_huffman_codes(node, prefix="", codebook={}):
    if node is not None:
        if node.symbol is not None:
            codebook[node.symbol] = prefix
        generate_huffman_codes(node.left, prefix + "0", codebook)
        generate_huffman_codes(node.right, prefix + "1", codebook)
    return codebook
# build huffman tree
huffman_tree = build_huffman_tree(symbol_counts)
huffman_codes = generate_huffman_codes(huffman_tree)
# measure the table size
huffman_table_size = sys.getsizeof(huffman_codes)
print(f"Huffman Table Size: {huffman_table_size} bytes")

# deal with symbols with no huffman code
def find_closest_symbol(symbol, valid_symbols):
    valid_symbols = np.array(list(valid_symbols), dtype=np.int16)  
    closest_index = np.abs(valid_symbols - symbol).argmin()  
    return valid_symbols[closest_index]
# encode
start_time = time.time()
encoded_data = []
for symbol in audio_samples:
    if symbol in huffman_codes:
        encoded_data.append(huffman_codes[symbol])
    else:
        closest_symbol = find_closest_symbol(symbol, huffman_codes.keys())
        encoded_data.append(huffman_codes.get(closest_symbol, "000000"))  

encoded_data = "".join(encoded_data)
encoding_time = time.time() - start_time
print("Huffman Encoding Time:", encoding_time, "seconds")

# convert encoded data into a bit array
bit_array = bitarray.bitarray(encoded_data)
compressed_file = "compressed_audio_random50.huff"
with open(compressed_file, "wb") as f:
    bit_array.tofile(f)


with open(compressed_file, "rb") as f:
    bit_array = bitarray.bitarray()
    bit_array.fromfile(f)

encoded_data = bit_array.to01()
# decode
def decode_huffman(encoded_data, huffman_tree):
    decoded_output = []
    node = huffman_tree
    for bit in encoded_data:
        node = node.left if bit == '0' else node.right
        if node.symbol is not None:
            decoded_output.append(node.symbol)
            node = huffman_tree
    return np.array(decoded_output, dtype=np.int16)

start_time = time.time()
decoded_samples = decode_huffman(encoded_data, huffman_tree)
decoding_time = time.time() - start_time
print(f"Huffman Decoding Time: {decoding_time} seconds")

# play for 5s
num_samples = 5 * sample_rate  
decoded_samples_short = decoded_samples[:num_samples]
sd.play(decoded_samples_short, samplerate=sample_rate)
sd.wait()

# Save the decoded audio for verification
wavfile.write("decoded_audio_5s_random50.wav", sample_rate, decoded_samples_short)

# compression ratio
original_bits = len(audio_samples) * 16
compressed_bits = len(encoded_data)
compression_ratio = 1 - (compressed_bits / original_bits)
print(f"Huffman Compression Ratio: {compression_ratio:.2%}")

# accuracy and check if decoded data matches the original input
if len(audio_samples) != len(decoded_samples):
    print(f"Original data length: {len(audio_samples)}, Decoded data length: {len(decoded_samples)}, mismatch detected, accuracy cannot be directly calculated!")
else:
    correct = np.sum(audio_samples == decoded_samples)
    accuracy = correct / len(audio_samples)
    print(f"ACCURACY: {accuracy * 100:.2f}%")

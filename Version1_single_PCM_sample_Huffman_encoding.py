import os
import wave
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import heapq
import time
import bitarray
import sounddevice as sd
import scipy.io.wavfile as wavfile
import sys 

#read WAV file and analyse
# current file path
current_dir = os.getcwd()

# all audio are stored in the audio folder
audio_dir = os.path.join(current_dir, "audio")


# real WAV files
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

    # Converts binary data to a numpy array
    samples = np.frombuffer(audio_data, dtype=np.int16)

    audio_samples.extend(samples)

audio_samples = np.array(audio_samples)

#the number of occurrences of PCM sample values
symbol_counts = Counter(audio_samples)

# convert to DataFrame
symbol_freq_df = pd.DataFrame(symbol_counts.items(), columns=["Symbol", "Frequency"]).sort_values(by="Frequency", ascending=False)

# Plot a histogram of audio sample values
# plt.figure(figsize=(10, 5))
# plt.hist(audio_samples, bins=100, color="blue", alpha=0.7, edgecolor="black")
# plt.title("Histogram of Audio Sample Values")
# plt.xlabel("PCM Sample Value")
# plt.ylabel("Times")
# plt.grid(True)
# plt.show()

# # Plot the occurence times of the top 100 most common PCM symbols
# plt.figure(figsize=(12, 6))
# plt.bar(symbol_freq_df["Symbol"].head(100), symbol_freq_df["Frequency"].head(100), color="orange", edgecolor="black")
# plt.xlabel("PCM Sample Value")
# plt.ylabel("Times")
# plt.title("Top 100 Most Common PCM Symbol Times")
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.show()

#Build the Huffman tree and generate Huffman codes
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

huffman_tree = build_huffman_tree(symbol_counts)
huffman_codes = generate_huffman_codes(huffman_tree)

### table Size
huffman_table_size = sys.getsizeof(huffman_codes)
print(f"Table Size: {huffman_table_size} bytes")

#Encoding and storage
start_time = time.time()
encoded_data = "".join(huffman_codes[symbol] for symbol in audio_samples)
encoding_time = time.time() - start_time
print("Encoding Time:", encoding_time, "seconds")

# convert to bitarray and store as .huff
bit_array = bitarray.bitarray(encoded_data)
compressed_file = "compressed_audio.huff"
with open(compressed_file, "wb") as f:
    bit_array.tofile(f)

#decoding 
# read the encoded file
with open(compressed_file, "rb") as f:
    bit_array = bitarray.bitarray()
    bit_array.fromfile(f)
encoded_data = bit_array.to01()

def decode_huffman(encoded_data, huffman_tree):
    decoded_output = []
    node = huffman_tree
    for bit in encoded_data:
        node = node.left if bit == '0' else node.right
        if node.symbol is not None:
            decoded_output.append(node.symbol)
            node = huffman_tree
    return np.array(decoded_output, dtype=np.int16)

# Measuring Huffman decoding time
start_time = time.time()
decoded_samples = decode_huffman(encoded_data, huffman_tree)
decoding_time = time.time() - start_time
print(f"Huffman Decoding Time: {decoding_time} seconds")

#play the first 5 seconds of decoded audio
sample_rate = 8000  
num_samples = 5 * sample_rate  
decoded_samples_short = decoded_samples[:num_samples]

# play and storage
sd.play(decoded_samples_short, samplerate=sample_rate)
sd.wait()
wavfile.write("decoded_audio_5s.wav", sample_rate, decoded_samples_short)

#compression ratio
original_bits = len(audio_samples) * 16  
compressed_bits = len(encoded_data)  
compression_ratio = 1 - (compressed_bits / original_bits)
print(f"Huffman Compression Ratio: {compression_ratio:.2%}")

#  check if the lengths are consistent
if len(audio_samples) != len(decoded_samples):
    print(f"Original data length: {len(audio_samples)}, Decoded data length: {len(decoded_samples)}, mismatch detected, accuracy cannot be directly calculated!")
else:
    # the number of matching samples
    correct = np.sum(audio_samples == decoded_samples)
    accuracy = correct / len(audio_samples)
    print(f"ACCURACY: {accuracy * 100:.2f}%")

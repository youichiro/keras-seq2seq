import keras
import pickle
import numpy as np

save_dir = 'tanaka_en_ja'
max_decoder_seq_length = 70

# Load models
encoder_model = keras.models.load_model(save_dir + '/encoder_model.h5')
encoder_model.load_weights(save_dir + '/encoder_model_weights.2.h5')
decoder_model = keras.models.load_model(save_dir + '/decoder_model.h5')
decoder_model.load_weights(save_dir + '/decoder_model_weights.2.h5')

# Load holds
with open(save_dir + '/holds.pkl', 'rb') as f:
    holds = pickle.load(f)
src_tokenizer = holds['src_tokenizer']
tgt_tokenizer = holds['tgt_tokenizer']
src_word2id = holds['src_word2id']
tgt_word2id = holds['tgt_word2id']
max_src = holds['max_src']
max_tgt = holds['max_tgt']
tgt_id2word = {v: k for k, v in tgt_word2id.items()}
num_decoder_tokens = max(tgt_word2id.values()) + 1

def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, tgt_word2id['bos']] = 1.

    stop_condition = False
    decoded_sentence = []
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = tgt_id2word[sampled_token_index]
        decoded_sentence.append(sampled_token)

        if (sampled_token == 'eos' or sampled_token == 'ignore' or len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        states_value = [h, c]

    return decoded_sentence

# Test
seq = 'i have a rule'
input_seq = src_tokenizer.texts_to_sequences([seq])
input_seq = [[np.pad(s, (0, max_src - len(s)), 'constant', constant_values=0) for s in input_seq]]
input_seq = keras.utils.to_categorical(input_seq)
decoded_sentence = decode_sequence(input_seq[0])
print(seq)
print(' '.join(decoded_sentence))

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import keras
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np
import pickle

# Files
src_file = 'datasets/train.en.1000'
tgt_file = 'datasets/train.ja.1000'
save_dir = 'tanaka_en_ja'
os.makedirs(save_dir, exist_ok=True)

# Hyperparameters
batch_size = 100
epochs = 3
latent_dim = 256

# Prepare data
src_data = [line.rstrip() for line in open(src_file, 'r')]
tgt_data = ['bos ' + line.rstrip() + ' eos' for line in open(tgt_file, 'r')]
tgt_input_data = [s.replace(' eos', '') for s in tgt_data]
tgt_ref_data = [s.replace('bos ', '') for s in tgt_data]
max_src = max([len(s) for s in src_data])
max_tgt = max([len(s) for s in tgt_data])

src_tokenizer = keras.preprocessing.text.Tokenizer()
src_tokenizer.fit_on_texts(src_data)
src_word2id = src_tokenizer.word_index
src_word2id['ignore'] = 0 
encoder_input_seq = src_tokenizer.texts_to_sequences(src_data)
encoder_input_seq = [np.pad(s, (0, max_src - len(s)), 'constant', constant_values=0) for s in encoder_input_seq]
encoder_input_data = to_categorical(encoder_input_seq)

tgt_tokenizer = keras.preprocessing.text.Tokenizer()
tgt_tokenizer.fit_on_texts(tgt_data)
tgt_word2id = tgt_tokenizer.word_index
tgt_word2id['ignore'] = 0

decoder_input_seq = tgt_tokenizer.texts_to_sequences(tgt_input_data)
decoder_input_seq = [np.pad(s, (0, max_tgt - len(s)), 'constant', constant_values=0) for s in decoder_input_seq]
decoder_input_data = to_categorical(decoder_input_seq)

decoder_target_seq = tgt_tokenizer.texts_to_sequences(tgt_ref_data)
decoder_target_seq = [np.pad(s, (0, max_tgt - len(s)), 'constant', constant_values=0) for s in decoder_target_seq]
decoder_target_data = to_categorical(decoder_target_seq)

# Save vocabs
vocabs = { 'src_word2id': src_word2id, 'tgt_word2id': tgt_word2id }
with open(save_dir + '/vocabs.pkl', 'wb') as f:
    pickle.dump(vocabs, f)


def build_and_save_predict_models(latent_dim, encoder_inputs, encoder_states, decoder_inputs, epoch):
    # Predict models
    encoder_model = Model(encoder_inputs, encoder_states)
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    # Save predict models
    encoder_model.save(save_dir + '/encoder_model.h5')
    encoder_model.save_weights(save_dir + '/encoder_model_weights.{}.h5'.format(epoch+1))
    decoder_model.save(save_dir + '/decoder_model.h5')
    decoder_model.save_weights(save_dir + '/decoder_model_weights.{}.h5'.format(epoch+1))


# CustomCallback
class SavePredictModelCallback(keras.callbacks.Callback):
    def __init__(self, latent_dim, encoder_inputs, encoder_states, decoder_inputs, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.encoder_inputs = encoder_inputs
        self.encoder_states = encoder_states
        self.decoder_inputs = decoder_inputs

    def on_epoch_end(self, epoch, logs={}):
        build_and_save_predict_models(self.latent_dim, self.encoder_inputs,
                                      self.encoder_states, self.decoder_inputs, epoch)
        print('Saved predict models.')

# Train model
encoder_inputs = Input(shape=(None, len(src_word2id)))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]
decoder_inputs = Input(shape=(None, len(tgt_word2id)))
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(len(tgt_word2id), activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# Run training
hist = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size, epochs=epochs, validation_split=0.2, verbose=1,
          callbacks=[SavePredictModelCallback(latent_dim, encoder_inputs, encoder_states, decoder_inputs)])

# Save history
open(save_dir + '/history.txt', 'w').write(str(hist.history) + '\n')

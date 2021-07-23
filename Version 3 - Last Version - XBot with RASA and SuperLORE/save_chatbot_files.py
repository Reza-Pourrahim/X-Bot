import json
import matplotlib.pyplot as plt

from xbot_model import XBotModel


# LORE X-BOT
data_file = open('chat_dataset/intents.json').read()
data = json.loads(data_file)

XBot_obj = XBotModel(verbose=True)
train_dataset, vocab_size, tokenizer, embedding_matrix = XBot_obj.prepare_train_dataset(data)
train_x = list(train_dataset[:, 0])
train_y = list(train_dataset[:, 1])


# create model
model = XBot_obj.create_model(train_x, train_y, embedding_matrix, vocab_size,
                              embedding_dim=100, lstm_out=15, dropout=0.5)

# compile and fit the model
mymodel = XBot_obj.compile_fit_model(model, train_x, train_y, epochs=200,
                                     batch_size=5,
                                     earlystopping_patience=10,
                                     validation_split=0.0,
                                     loss='categorical_crossentropy')



# plot the Train and Validation loss
plt.plot(mymodel['loss'], label='Train')
# plt.plot(mymodel['val_loss'], label='Val')
plt.xlabel('Epochs')
plt.ylabel('Cross-Entropy')
plt.legend()
plt.savefig('chatbot_model_files/loss.png')
plt.show()

# plot the Train and Validation accuracy
plt.plot(mymodel['accuracy'], label='Train')
# plt.plot(mymodel['val_accuracy'], label='Val')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('chatbot_model_files/accuracy.png')
plt.show()

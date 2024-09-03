from importing_librarues.lib import train_test_split,Sequential,Conv2D,MaxPooling2D,Flatten,Dense,Dropout,ImageDataGenerator,plt,cv2,tf,np,confusion_matrix,sns
from Dataset_loading_and_preprocessing.loading_datasets import train_df,images,num_classes,unique_labels,y_classes

x_temp, x_test, y_temp, y_test = train_test_split(images, train_df.categories.values, test_size=0.2, random_state=8, shuffle=True)

x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=0.25, random_state=8, shuffle=True)

#Normalize the data
x_train = x_train / 255.0
x_test = x_test / 255.0
x_val = x_val / 255.0

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_val = x_val.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')
y_val = y_val.astype('float32')

input_shape = (224, 224, 3)

def create_model(input_shape):
    model = Sequential()

    model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D())

    model.add(Conv2D(32, (3,3), 1, activation='relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(64, (3,3), 1, activation='relu'))
    model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))

    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    return model

model = create_model(input_shape)
model.summary()

#Model Training

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
)

train_generator = datagen.flow(x_train, y_train, batch_size=32)

history = model.fit(train_generator, epochs=50, validation_data=(x_val, y_val))

result = model.evaluate(x_test, y_test)
print(result)

fig = plt.figure()
plt.plot(history.history['loss'], color='teal', label='loss')
plt.plot(history.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()

fig = plt.figure()
plt.plot(history.history['accuracy'], color='teal', label='accuracy')
plt.plot(history.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()

#Testing on an image

path = "" #enter path of an image
img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
resized_img = tf.image.resize(img, (224, 224))

y_pred = model.predict(np.expand_dims(resized_img/255, 0))
y_class = np.argmax(y_pred[0])
print(unique_labels[y_class])

y_pred = model.predict(x_test)
y_classes = [np.argmax(el) for el in y_pred]

# Confussion Matrix
cm = confusion_matrix(y_test, y_classes)
plt.rcParams['font.size'] = 16
plt.figure(figsize=(20, 15))
sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels, fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

plt.rcParams['font.size'] = 12

accuracy = result[1]

metrics = [history.history['accuracy'][-1], accuracy]
labels = ['Training Accuracy', 'Actual Accuracy']

metrics_percentages = [x * 100 for x in metrics]

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))

plt.rcParams['font.size'] = 10
ax.bar(labels, metrics_percentages)

for i in range(len(metrics_percentages)):
    ax.annotate(str(round(metrics_percentages[i], 1))+"%", xy=(labels[i], metrics_percentages[i]), ha='center', va='bottom')

ax.set_ylim([0, 100])
ax.set_title('Classification Metrics')
ax.tick_params(axis='x', labelrotation=45)

plt.rcParams['font.size'] = 12

plt.show()

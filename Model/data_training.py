import os  
import numpy as np 
import cv2 
from keras.utils import to_categorical
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.layers import Input, Dense, Dropout, MaxPooling2D, Reshape
from keras.models import Model,Sequential

from data_augmentation import func

current_directory = os.getcwd()
folder_path = os.path.join(current_directory, 'Asanas')

def train_asanawise():
	folder_path = os.path.join(current_directory, 'New_data')
	csv_folder_path = os.path.join(folder_path, 'csvs')			# csvs

	for csv_file in os.listdir(csv_folder_path):			# e1
		if not csv_file.split('.')[-1]=="csv":
			continue
		csv_file_path = os.path.join(csv_folder_path,csv_file)
		df = pd.read_csv(csv_file_path)

		X = df.iloc[:, :-1]
		y = df.iloc[:, -1]
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

		n_inter_poses = len(np.unique(y))

		model = Sequential([
			Dense(512, activation='relu', input_shape=(99,)),
			Dropout(0.3),
			Dense(256, activation='relu'),
			Dropout(0.3),
			Dense(128, activation='relu'),
			Dense(64, activation='relu'),
			Dropout(0.3),
			Dense(32, activation='relu'),
			Dense(16, activation='relu'),
			Dense(n_inter_poses, activation='softmax')
		])
		print(model.summary())
		model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
		history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=140)
		loss, accuracy = model.evaluate(X_test, y_test)
		print(loss,accuracy)
		model_path = os.path.join(folder_path,'models')
		model.save(os.path.join(model_path,f'{csv_file.split(".")[0]}.h5'))

		y_pred = model.predict(X_test)
		y_pred = [np.argmax(i) for i in y_pred]

		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('Model Loss')
		plt.ylabel('Loss')
		plt.xlabel('Epochs')
		plt.legend(['Train', 'Validation'], loc='upper left')
		# plt.show()
		plt.savefig(os.path.join(current_directory,'Plots','Asana_wise',f'Loss_{csv_file.split(".")[0]}.png'))

		cm = confusion_matrix(y_test,y_pred)
		print(cm)
		disp = ConfusionMatrixDisplay(confusion_matrix=cm)
		disp.plot()
		# plt.show()
		plt.savefig(os.path.join(current_directory,'Plots','Asana_wise',f'Cm_{csv_file.split(".")[0]}.png'))

def train_trainerwise():
	folder_path = os.path.join(current_directory, 'New_data')
	csv_folder_path = os.path.join(folder_path, 'csvs')			# csvs

	for trainer in os.listdir(csv_folder_path):								# trainer_name
		if trainer.split('.')[-1]=="csv":
			continue
		trainer_path = os.path.join(csv_folder_path,trainer)
		print(trainer)
		for csv_file in os.listdir(trainer_path):			# e1
			csv_file_path = os.path.join(trainer_path,csv_file)
			df = pd.read_csv(csv_file_path)

			X = df.iloc[:, :-1]
			y = df.iloc[:, -1]
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

			n_inter_poses = len(np.unique(y))

			model = Sequential([
				Dense(512, activation='relu', input_shape=(99,)),		# 99 or 115
				Dropout(0.3),
				Dense(256, activation='relu'),
				Dropout(0.3),
				Dense(128, activation='relu'),
				Dense(64, activation='relu'),
				Dropout(0.3),
				Dense(32, activation='relu'),
				Dense(16, activation='relu'),
				Dense(n_inter_poses, activation='softmax')
			])
			print(model.summary())
			model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
			history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=140)
			loss, accuracy = model.evaluate(X_test, y_test)
			print(loss,accuracy)
			# model_path = os.path.join(folder_path,'models')
			os.makedirs(os.path.join(current_directory,'New_data','models',f'{trainer}'),exist_ok=True)
			model_path = os.path.join(current_directory,'New_data','models',f'{trainer}')
			model.save(os.path.join(model_path,f'{csv_file.split(".")[0]}.h5'))

			y_pred = model.predict(X_test)
			y_pred = [np.argmax(i) for i in y_pred]

			plt.plot(history.history['loss'])
			plt.plot(history.history['val_loss'])
			plt.title('Model Loss')
			plt.ylabel('Loss')
			plt.xlabel('Epochs')
			plt.legend(['Train', 'Validation'], loc='upper left')
			# plt.show()
			plt.savefig(os.path.join(current_directory,'Plots','Trainer_wise',f'Loss_{trainer}_{csv_file.split(".")[0]}.png'))

			cm = confusion_matrix(y_test,y_pred)
			print(cm)
			disp = ConfusionMatrixDisplay(confusion_matrix=cm)
			disp.plot()
			plt.savefig(os.path.join(current_directory,'Plots','Trainer_wise',f'Cm_{trainer}_{csv_file.split(".")[0]}.png'))
			# plt.show()

train_asanawise()
# train_trainerwise()
		

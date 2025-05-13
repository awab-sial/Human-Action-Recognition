import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    QLabel,
    QMenuBar,
    QAction,
    QMessageBox,
)
from PyQt5.QtGui import QPixmap, QPalette, QBrush
from PyQt5.QtCore import Qt
import tensorflow as tf
from keras import models
import matplotlib.pyplot as plt


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Window settings
        self.setWindowTitle("AI Human Action Recognition")
        self.setGeometry(200, 200, 800, 600)

        # Set the background image
        self.set_background_image("img.jpg")

        # Central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # Add nav bar
        self.create_menu_bar()

        # Label for displaying uploaded file
        self.file_label = QLabel("Upload an image or video for detection")
        self.file_label.setAlignment(Qt.AlignCenter)
        self.file_label.setStyleSheet(
            """
            color: #f0f0f0;
            font-size: 16px;
            background-color: rgba(50, 50, 50, 0.7);
            border: 2px solid #f0f0f0;
            border-radius: 10px;
            padding: 8px;
            """
        )
        self.file_label.setMaximumWidth(400)
        self.layout.addWidget(self.file_label, alignment=Qt.AlignCenter)

        # Add Buttons
        self.add_buttons()

        # Placeholder for uploaded file path
        self.file_path = None

        # Load the pretrained model
        self.model = models.load_model("/Users/Desktop/human_action_model.h5")

        # Initialize ImageDataGenerator for class indices
        self.train_dir = "/Users/fawadnaveed/Desktop/SEProject/Structured/train"
        self.train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255.0)
        self.train_data = self.train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(128, 128),
            batch_size=32,
            class_mode='categorical'
        )

    def set_background_image(self, image_path):
        """Sets a background image for the main window."""
        palette = QPalette()
        pixmap = QPixmap(image_path)
        palette.setBrush(QPalette.Background, QBrush(pixmap))
        self.setPalette(palette)

    def create_menu_bar(self):
        """Creates a styled navigation bar."""
        menu_bar = QMenuBar(self)
        menu_bar.setStyleSheet(
            """
            QMenuBar {
                background-color: #4a90e2;
                color: white;
                font-size: 16px;
                padding: 5px;
            }
            QMenuBar::item {
                background-color: transparent;
                padding: 5px 10px;
            }
            QMenuBar::item:selected {
                background-color: #357abd;
            }
            QMenu {
                background-color: #4a90e2;
                color: white;
                margin: 2px;
            }
            QMenu::item:selected {
                background-color: #357abd;
            }
            """
        )

        # File menu
        file_menu = menu_bar.addMenu("File")
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Help menu
        help_menu = menu_bar.addMenu("Help")
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

        self.setMenuBar(menu_bar)

    def add_buttons(self):
        """Adds styled buttons."""
        button_style = """
        QPushButton {
            background-color: rgba(74, 144, 226, 0.8);
            color: white;
            font-size: 16px;
            border-radius: 10px;
            padding: 10px;
            border: 2px solid rgba(53, 122, 189, 0.8);
        }
        QPushButton:hover {
            background-color: rgba(53, 122, 189, 0.85);
        }
        QPushButton:pressed {
            background-color: rgba(40, 86, 138, 0.9);
        }
        """
        # Upload Button
        upload_button = QPushButton("Upload")
        upload_button.setStyleSheet(button_style)
        upload_button.clicked.connect(self.upload_file)
        self.layout.addWidget(upload_button)

        # Detect Button
        detect_button = QPushButton("Detect")
        detect_button.setStyleSheet(button_style)
        detect_button.clicked.connect(self.detect_action)
        self.layout.addWidget(detect_button)

        # Exit Button
        exit_button = QPushButton("Exit")
        exit_button.setStyleSheet(button_style)
        exit_button.clicked.connect(self.close)
        self.layout.addWidget(exit_button)

    def upload_file(self):
        """Handles file uploads."""
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_filter = "Image Files (*.jpg *.png)"
        file_path, _ = QFileDialog.getOpenFileName(self, "Upload File", "", file_filter, options=options)

        if file_path:
            self.file_path = file_path
            file_name = os.path.basename(file_path)

            if file_path.endswith((".jpg", ".png")):
                pixmap = QPixmap(file_path)
                self.file_label.setPixmap(pixmap.scaled(400, 300, Qt.KeepAspectRatio))
            else:
                self.file_label.setText(f"Uploaded: {file_name}")

    def detect_action(self):
        """Detects the action from the uploaded file."""
        if self.file_path:
            predicted_action = self.predict_action(self.file_path)
            self.file_label.setText(f"Predicted Action: {predicted_action}")

            image = cv2.imread(self.file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.imshow(image)
            plt.title(f"Predicted: {predicted_action}")
            plt.axis('off')
            plt.show()
        else:
            QMessageBox.warning(self, "Warning", "Please upload an image before detection!")

    def predict_action(self, image_path):
        """Predict the action for the uploaded image."""
        image = cv2.imread(image_path)
        image = cv2.resize(image, (128, 128))
        image = np.expand_dims(image, axis=0) / 255.0

        prediction = self.model.predict(image)
        predicted_class = list(self.train_data.class_indices.keys())[np.argmax(prediction)]
        return predicted_class

    def show_about(self):
        """Shows an About dialog."""
        QMessageBox.information(
            self,
            "About",
            "This is a PyQt-based GUI for an AI Human Action Recognition model.\n\n"
            "1. Upload: Uploads an image or video.\n"
            "2. Detect: Runs the AI model.\n"
            "3. Exit: Closes the application.",
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec_())

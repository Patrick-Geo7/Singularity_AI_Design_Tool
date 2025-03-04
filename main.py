from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QLabel, QFrame
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from ui_main import Ui_MainWindow  # Import UI class
import torch
import os
import traceback
import matplotlib.pyplot as plt
from train import BathroomPlacementModel
from test import visualize_prediction


class MainApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)  # Initialize UI
        self.pushButton.clicked.connect(self.generate_prediction)  # Connect button

    def generate_prediction(self):
        try:
            # Get width and length from input fields
            width = float(self.textEdit_2.toPlainText())
            length = float(self.textEdit.toPlainText())
            if width <= 0 or length <= 0:
                raise ValueError("Dimensions must be positive numbers.")

            room_dims = {"width": width, "length": length}
            print(f"Room dimensions set to: {room_dims}")

            # Load model
            print("Loading trained model...")
            model_path = "best_model.pth"
            if not os.path.exists(model_path):
                print(f"Error: Model file not found at {model_path}")
                return

            checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
            input_dim, output_dim = 8, 15
            model = BathroomPlacementModel(input_dim, output_dim)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            # Generate random inputs
            random_features = torch.rand(1, input_dim - 2) * 10
            user_input = torch.tensor([[width, length]])
            random_input = torch.cat((user_input, random_features), dim=1)

            # Get prediction
            with torch.no_grad():
                predictions = model(random_input).numpy()
            print("Predictions generated successfully")

            # Visualize prediction
            image_path = "prediction_plot.png"
            visualize_prediction(predictions, room_dims)  # Saves the plot to file

            # Display the plot in UI
            self.load_plot_to_frame(image_path)
            self.load_plot_to_label(image_path)

        except ValueError:
            QtWidgets.QMessageBox.critical(self, "Error", "Please enter valid numeric values for width and length.")
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Error", "An error occurred while generating the prediction.")

    def load_plot_to_frame(self, image_path):
        """Load and display the saved plot inside the QFrame."""
        frame = self.findChild(QFrame, "frame_4")
        if frame is None:
            print("Error: QFrame for displaying image not found. Check UI object name.")
            return

        # Remove previous QLabel if exists
        if hasattr(self, "frame_image_label"):
            self.frame_image_label.setParent(None)

        # Create QLabel inside the QFrame
        self.frame_image_label = QLabel(frame)
        self.frame_image_label.setGeometry(0, 0, frame.width(), frame.height())
        self.frame_image_label.setAlignment(Qt.AlignCenter)
        self.frame_image_label.setScaledContents(True)

        # Load and set the image
        pixmap = QPixmap(image_path)
        self.frame_image_label.setPixmap(pixmap)
        self.frame_image_label.show()

    def load_plot_to_label(self, image_path):
        """Load the saved plot and display it in QLabel (label_6)."""
        label = self.findChild(QLabel, "label_6")
        if label is None:
            print("Error: QLabel 'label_6' not found. Check UI object name.")
            return

        # Load and set the image
        pixmap = QPixmap(image_path)
        label.setPixmap(pixmap)
        label.setScaledContents(True)


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainApp()
    window.show()
    app.exec_()

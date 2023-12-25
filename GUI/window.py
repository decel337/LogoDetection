import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class ImageProcessor:
    def __init__(self, root):
        self.root = root
        self.root.title("Logo Detection")
        self.root.geometry("800x600")

        self.left_frame = tk.Frame(root, bd=5, relief="groove")
        self.left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.right_frame = tk.Frame(root, bd=5, relief="groove")
        self.right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        self.image_label = tk.Label(self.left_frame, text="Select image", font=("Helvetica", 12), bg="lightgray")
        self.image_label.pack(expand=True, fill="both")

        self.load_button = tk.Button(self.left_frame, text="Download image", command=self.load_image)
        self.load_button.pack(side=tk.BOTTOM, pady=10)

        self.processed_image_label = tk.Label(self.right_frame, text="Logo detected", font=("Helvetica", 12), bg="lightgray")
        self.processed_image_label.pack(expand=True, fill="both")

        self.current_image = None
        self.current_image_path = None

        self.start_button = tk.Button(root, text="DETECT", command=self.start_processing)
        self.start_button.grid(row=1, column=0, columnspan=2, pady=10)

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

    def load_image(self):
        file_path = filedialog.askopenfilename(title="Select image",
                                                filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.gif")])
        if file_path:
            self.display_image(file_path)

    def display_image(self, file_path):
        if self.current_image:
            self.current_image.destroy()

        self.image_label.config(text="")

        image = Image.open(file_path)

        max_width = self.left_frame.winfo_width() - 20
        max_height = self.left_frame.winfo_height() - 20

        if image.width > max_width or image.height > max_height:
            image.thumbnail((max_width, max_height), Image.ANTIALIAS)

        photo = ImageTk.PhotoImage(image)

        self.current_image = tk.Label(self.left_frame, image=photo, padx=10, pady=10)
        self.current_image.image = photo
        self.current_image.pack(expand=True, fill="both")

        self.image_label.pack_forget()
        self.current_image_path = file_path

    def start_processing(self):
        if self.current_image_path:

            self.processed_image_label.config(image=self.current_image.image)
            self.processed_image_label.image = self.current_image.image

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessor(root)
    root.mainloop()

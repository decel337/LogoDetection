import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
from logodetection import detect

class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processor")
        self.root.geometry("600x400")

        self.left_image = None
        self.right_image = None
        self.progress = False

        self.create_widgets()
        self.progress_bar = ttk.Progressbar(root, orient="horizontal", mode="indeterminate")

        self.root.bind("<Configure>", self.update_image_sizes)

    def create_widgets(self):
        image_frame = tk.Frame(self.root)
        image_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.static_frame_left = tk.Canvas(image_frame, bg="white", highlightthickness=1, highlightbackground="black",
                                      width=300, height=self.root.winfo_height() - 50)
        self.static_frame_left.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        self.static_frame_right = tk.Canvas(image_frame, bg="white", highlightthickness=1, highlightbackground="black",
                                       width=300, height=self.root.winfo_height() - 50)
        self.static_frame_right.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")

        self.load_button = tk.Button(image_frame, text="Select image", command=self.load_image_left)
        self.load_button.grid(row=2, column=0, pady=10, padx=(0, 50), sticky="ew")

        self.save_button = tk.Button(image_frame, text="Save", command=self.save_image)
        self.save_button.grid(row=2, column=1, pady=10, padx=(50, 0), sticky="ew")

        self.start_button = tk.Button(image_frame, text="Detect", command=self.process_image)
        self.start_button.grid(row=3, column=0, columnspan=2, pady=10)

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        image_frame.columnconfigure(0, weight=1)
        image_frame.columnconfigure(1, weight=1)
        image_frame.rowconfigure(0, weight=0)
        image_frame.rowconfigure(1, weight=1)

    def load_image_left(self):
        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.left_image = Image.open(file_path)
            self.display_image(self.left_image, self.static_frame_left)

    def process_image(self):
        if self.left_image:
            self.disable_buttons()
            self.progress = True
            self.root.after(100, self.process)

            self.progress_bar.grid(row=4, column=0, columnspan=2, pady=10)
            self.progress_bar.start()

    def save_image(self):
        if self.right_image:
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("Images", "*.png;*.jpg;*.jpeg")])
            if file_path:
                self.right_image.save(file_path)

    def process(self):
        try:
            image_for_process = self.left_image.copy()
            processed_image = detect(image_for_process)
            self.right_image = processed_image
            self.display_image(self.right_image, self.static_frame_right)
        except Exception as e:
            messagebox.showerror("Error", f"{str(e)}")

        self.progress_bar.stop()
        self.progress_bar.grid_forget()
        self.enable_buttons()
        self.progress = False

    def display_image(self, image, canvas):
        window_width = self.root.winfo_width() // 2
        window_height = self.root.winfo_height() - 50

        scale_factor_width = window_width / image.width
        scale_factor_height = window_height / image.height

        scale_factor = min(scale_factor_width, scale_factor_height)

        new_width = int(image.width * scale_factor)
        new_height = int(image.height * scale_factor)
        image = image.resize((new_width, new_height), Image.ANTIALIAS)

        photo = ImageTk.PhotoImage(image)

        canvas.delete("all")
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        canvas.image = photo

    def disable_buttons(self):
        self.load_button.configure(state='disabled')
        self.start_button.configure(state='disabled')
        self.save_button.configure(state='disabled')

    def enable_buttons(self):
        self.load_button.configure(state='normal')
        self.start_button.configure(state='normal')
        self.save_button.configure(state='normal')

    def update_image_sizes(self, event):
        if not self.progress:
            if self.left_image:
                self.display_image(self.left_image, self.static_frame_left)
            if self.right_image:
                self.display_image(self.right_image, self.static_frame_right)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()

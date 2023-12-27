import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox, simpledialog
from PIL import Image, ImageTk
from logodetection import detect


class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processor")
        self.root.geometry("600x400")

        self.menu_bar = tk.Menu(self.root)
        self.root.config(menu=self.menu_bar)

        file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Edit", menu=file_menu)
        file_menu.add_command(label="Clear", command=self.clear)
        self.menu_bar.add_cascade(label="Settings", menu=file_menu)
        file_menu.add_command(label="Threshold", command=self.thresh)

        self.left_images = []
        self.right_images = []
        self.progress = False
        self.selected_left_index = 0
        self.selected_right_index = 0
        self.threshold = 0

        self.create_widgets()
        self.progress_bar = ttk.Progressbar(root, orient="horizontal", mode="indeterminate")

        self.root.bind("<Configure>", self.update_image_sizes)

    def create_widgets(self):
        image_frame = tk.Frame(self.root)
        image_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.static_frame_left = tk.Canvas(image_frame, bg="white", highlightthickness=1, highlightbackground="black",
                                           width=300, height=self.root.winfo_height() - 50)
        self.static_frame_left.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        self.left_scale = tk.Scale(image_frame, orient="horizontal", command=self.left_scroll_with_scale,
                                   showvalue=False, length=400)
        self.left_scale.grid(row=0, column=0, pady=5)
        self.left_scale.config(from_=0, to=0, resolution=1)
        self.static_frame_right = tk.Canvas(image_frame, bg="white", highlightthickness=1, highlightbackground="black",
                                            width=300, height=self.root.winfo_height() - 50)
        self.static_frame_right.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
        self.right_scale = tk.Scale(image_frame, orient="horizontal", command=self.right_scroll_with_scale, showvalue=False,
                                   length=400)
        self.right_scale.grid(row=0, column=1, pady=5)
        self.right_scale.config(from_=0, to=0, resolution=1)
        self.load_button = tk.Button(image_frame, text="Select image", command=self.load_image_left)
        self.load_button.grid(row=2, column=0, pady=10, padx=(0, 50), sticky="ew")

        self.load_folder_button = tk.Button(image_frame, text="Select folder", command=self.load_folder)
        self.load_folder_button.grid(row=3, column=0, pady=10, padx=(0, 50), sticky="ew")

        self.save_button = tk.Button(image_frame, text="Save", command=self.save_image)
        self.save_button.grid(row=2, column=1, pady=10, padx=(50, 0), sticky="ew")

        self.operate_button = tk.Button(image_frame, text="Split", command=self.process_image_split)
        self.operate_button.grid(row=3, column=1, pady=10, padx=(50, 0), sticky="ew")

        self.start_button = tk.Button(image_frame, text="Detect", command=self.process_image)
        self.start_button.grid(row=4, column=0, columnspan=2, pady=10)

        self.label1 = tk.Label(image_frame, text=f"Threshold {self.threshold}")
        self.label1.grid(row=4, column=0, pady=10)

        self.label2 = tk.Label(image_frame)
        self.label2.grid(row=4, column=1, pady=10)

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        image_frame.columnconfigure(0, weight=1)
        image_frame.columnconfigure(1, weight=1)
        image_frame.rowconfigure(0, weight=0)
        image_frame.rowconfigure(1, weight=1)

    def load_image_left(self):
        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.left_images = []
            self.selected_left_index = 0
            self.left_scale.config(from_=0, to=0, resolution=1)
            self.left_images.append(Image.open(file_path))
            self.display_image(self.left_images[self.selected_left_index], self.static_frame_left)

    def save_image(self):
        if self.right_images:
            file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                     filetypes=[("Images", "*.png;*.jpg;*.jpeg")])
            if file_path:
                self.right_images[self.selected_right_index].save(file_path)
    def process_image(self):
        if self.left_images:
            self.disable_buttons()
            self.progress = True
            self.progress_bar.grid(row=4, column=0, columnspan=2, pady=10)
            self.progress_bar.start()
            self.root.after(100, self.process(False))

    def process_image_split(self):
        if self.left_images:
            self.disable_buttons()
            self.progress = True
            self.progress_bar.grid(row=4, column=0, columnspan=2, pady=10)
            self.progress_bar.start()
            self.root.after(100, self.process(True))
    def process(self, isSplit = False):
        try:
            image_for_process = self.left_images[self.selected_left_index].copy()
            processed_images, time = detect(image_for_process, isSplit, self.threshold)
            self.right_images = processed_images
            self.right_scale.config(from_=0, to=len(self.right_images) - 1, resolution=1)
            self.selected_right_index = 0
            self.display_image(self.right_images[self.selected_right_index], self.static_frame_right)
            if isSplit:
                self.operate_button.config(text="Merge", command=self.process_image)
            else:
                self.operate_button.config(text="Split", command=self.process_image_split)
            self.label2.configure(text='Processed image in {:.1f}sec'.format(time))
        except Exception as e:
            messagebox.showerror("Error", f"{str(e)}")
        finally:
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
        self.operate_button.configure(state='disabled')
        self.load_folder_button.configure(state='disabled')

    def enable_buttons(self):
        self.load_button.configure(state='normal')
        self.start_button.configure(state='normal')
        self.save_button.configure(state='normal')
        self.operate_button.configure(state='normal')
        self.load_folder_button.configure(state='normal')

    def update_image_sizes(self, event):
        if not self.progress:
            if self.left_images:
                self.display_image(self.left_images[self.selected_left_index], self.static_frame_left)
            if self.right_images:
                self.display_image(self.right_images[self.selected_right_index], self.static_frame_right)

    def clear(self):
        self.left_images = []
        self.right_images = []
        self.selected_right_index = 0
        self.selected_left_index = 0
        self.left_scale.config(from_=0, to=0, resolution=1)
        self.right_scale.config(from_=0, to=0, resolution=1)

        self.static_frame_left.delete("all")
        self.static_frame_right.delete("all")

    def thresh(self):
        value = simpledialog.askfloat("Set Value", "Enter a value between 0 and 1:", initialvalue=self.threshold,
                                      minvalue=0, maxvalue=1)

        if value is not None:
            self.threshold = value
            self.label1.configure(text=f"Threshold {self.threshold}")
    def load_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
            image_extensions = [".png", ".jpg", ".jpeg"]
            image_files = [file for file in files if any(file.lower().endswith(ext) for ext in image_extensions)]
            self.left_images = []
            for image_file in image_files:
                image_path = os.path.join(folder_path, image_file)
                try:
                    image = Image.open(image_path)
                    self.left_images.append(image)
                except Exception as e:
                    messagebox.showinfo("Can't open image", str(e))
            self.selected_left_index = 0
            self.display_image(self.left_images[self.selected_left_index], self.static_frame_left)
            self.left_scale.config(from_=0, to=len(self.left_images) - 1, resolution=1)

    def left_scroll_with_scale(self, value):
        try:
            self.selected_left_index = int(value)
            self.display_image(self.left_images[self.selected_left_index], self.static_frame_left)
        except ValueError:
            pass

    def right_scroll_with_scale(self, value):
        try:
            self.selected_right_index = int(value)
            self.display_image(self.right_images[self.selected_right_index], self.static_frame_right)
        except ValueError:
            pass


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()

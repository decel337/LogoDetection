import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class ImageProcessor:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processor")
        self.root.geometry("800x600")  # Исходный размер окна

        # Создаем две рамки для левой и правой частей окна
        self.left_frame = tk.Frame(root, bd=5, relief="groove")
        self.left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.right_frame = tk.Frame(root, bd=5, relief="groove")
        self.right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # Левая часть: ячейка для отображения выбранного изображения
        self.image_label = tk.Label(self.left_frame, text="Выберите изображение", font=("Helvetica", 12), bg="lightgray")
        self.image_label.pack(expand=True, fill="both")

        # Правая часть: ячейка для отображения обработанного изображения
        self.processed_image_label = tk.Label(self.right_frame, text="Обработанное изображение", font=("Helvetica", 12), bg="lightgray")
        self.processed_image_label.pack(expand=True, fill="both")

        # Переменные для отслеживания текущего изображения и его пути
        self.current_image = None
        self.current_image_path = None

        # Кнопка Загрузить изображение
        self.load_button = tk.Button(root, text="Загрузить изображение", command=self.load_image)
        self.load_button.grid(row=1, column=0, pady=10)

        # Кнопка СТАРТ
        self.start_button = tk.Button(root, text="СТАРТ", command=self.start_processing)
        self.start_button.grid(row=2, column=0, columnspan=2, pady=10)

        # Кнопка Скачать изображение
        self.download_button = tk.Button(root, text="Скачать изображение", command=self.download_image)
        self.download_button.grid(row=1, column=1, pady=10)

        # Растягиваем строки и столбцы, чтобы они заполняли доступное пространство
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

    def load_image(self):
        file_path = filedialog.askopenfilename(title="Выберите изображение",
                                                filetypes=[("Изображения", "*.png;*.jpg;*.jpeg;*.gif")])
        if file_path:
            self.display_image(file_path)

    def display_image(self, file_path):
        if self.current_image:
            self.current_image.destroy()

        self.image_label.config(text="")  # Скрываем текст

        image = Image.open(file_path)

        # Проверяем размеры изображения
        max_width = self.left_frame.winfo_width() - 20
        max_height = self.left_frame.winfo_height() - 20

        if image.width > max_width or image.height > max_height:
            # Если изображение больше максимальных размеров, масштабируем его
            image.thumbnail((max_width, max_height), Image.ANTIALIAS)

        photo = ImageTk.PhotoImage(image)

        # Отображаем изображение в левой части
        self.current_image = tk.Label(self.left_frame, image=photo, padx=10, pady=10)
        self.current_image.image = photo
        self.current_image.pack(expand=True, fill="both")

        self.image_label.pack_forget()  # Убираем Label с текстом
        self.current_image_path = file_path

    def start_processing(self):
        if self.current_image_path:
            # Отображаем обработанное изображение в правой части
            processed_image = Image.open(self.current_image_path)
            processed_image.thumbnail((self.left_frame.winfo_width() - 20, self.left_frame.winfo_height() - 20), Image.ANTIALIAS)
            processed_photo = ImageTk.PhotoImage(processed_image)

            self.processed_image_label.config(image=processed_photo)
            self.processed_image_label.image = processed_photo
            self.processed_image_label.pack(expand=True, fill="both")

    def download_image(self):
        if self.processed_image_label.image:
            # Сохраняем обработанное изображение
            file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                        filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
            if file_path:
                processed_image = Image.open(self.current_image_path)
                processed_image.thumbnail((self.left_frame.winfo_width() - 20, self.left_frame.winfo_height() - 20), Image.ANTIALIAS)
                processed_image.save(file_path)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessor(root)
    root.mainloop()

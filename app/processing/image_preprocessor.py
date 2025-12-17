import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import os
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    def __init__(self):
        self.min_width = 800
        self.min_height = 600

    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Загрузить изображение"""
        try:
            if not os.path.exists(image_path):
                logger.error(f"Image not found: {image_path}")
                return None

            # Пытаемся загрузить через OpenCV
            image = cv2.imread(image_path)
            if image is None:
                # Пробуем через PIL
                pil_image = Image.open(image_path)
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            return image
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            return None

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Основная предобработка изображения"""
        try:
            # Конвертируем в оттенки серого
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Увеличиваем контраст
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            contrast_enhanced = clahe.apply(gray)

            # Бинаризация с адаптивным порогом
            binary = cv2.adaptiveThreshold(
                contrast_enhanced, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )

            # Убираем шум
            denoised = cv2.medianBlur(binary, 3)

            # Улучшаем резкость
            kernel = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]])
            sharpened = cv2.filter2D(denoised, -1, kernel)

            return sharpened

        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            return image

    def deskew_image(self, image: np.ndarray) -> np.ndarray:
        """Выравнивание изображения (дескри)"""
        try:
            # Находим контуры текста
            coords = np.column_stack(np.where(image > 0))

            if len(coords) < 10:
                return image  # Недостаточно точек для вычисления угла

            # Вычисляем угол поворота
            angle = cv2.minAreaRect(coords)[-1]

            if angle < -45:
                angle = 90 + angle

            # Поворачиваем изображение
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)

            if abs(angle) > 1:  # Поворачиваем только если угол значительный
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(
                    image, M, (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE
                )
                return rotated

            return image

        except Exception as e:
            logger.error(f"Error deskewing image: {str(e)}")
            return image

    def segment_into_works(self, image: np.ndarray, min_area: int = 50000) -> List[np.ndarray]:
        """Сегментировать изображение на отдельные работы"""
        try:
            # Находим контуры
            contours, _ = cv2.findContours(
                image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            works = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    x, y, w, h = cv2.boundingRect(contour)

                    # Вырезаем область работы
                    work = image[y:y + h, x:x + w]

                    # Добавляем отступы
                    padding = 20
                    work_padded = cv2.copyMakeBorder(
                        work, padding, padding, padding, padding,
                        cv2.BORDER_CONSTANT, value=255
                    )

                    works.append(work_padded)

            # Сортируем сверху вниз
            works.sort(key=lambda w: w.shape[0])

            return works

        except Exception as e:
            logger.error(f"Error segmenting image: {str(e)}")
            return [image]

    def extract_text_regions(self, image: np.ndarray) -> List[dict]:
        """Выделить области текста (ROI)"""
        try:
            # Морфологические операции для выделения текстовых блоков
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 10))
            dilated = cv2.dilate(image, kernel, iterations=2)

            contours, _ = cv2.findContours(
                dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)

                # Фильтруем маленькие области
                if w > 50 and h > 20:
                    region = image[y:y + h, x:x + w]

                    regions.append({
                        "bbox": (x, y, w, h),
                        "image": region,
                        "area": w * h
                    })

            # Сортируем по вертикальной позиции
            regions.sort(key=lambda r: r["bbox"][1])

            return regions

        except Exception as e:
            logger.error(f"Error extracting text regions: {str(e)}")
            return []

    def save_processed_image(self, image: np.ndarray, output_path: str) -> bool:
        """Сохранить обработанное изображение"""
        try:
            # Создаем директорию если нет
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Сохраняем
            cv2.imwrite(output_path, image)
            logger.info(f"Saved processed image to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving image {output_path}: {str(e)}")
            return False

    def create_thumbnail(self, image: np.ndarray, max_size: Tuple[int, int] = (300, 300)) -> np.ndarray:
        """Создать миниатюру изображения"""
        try:
            h, w = image.shape[:2]

            # Вычисляем новые размеры с сохранением пропорций
            if w > h:
                new_w = max_size[0]
                new_h = int(h * (max_size[0] / w))
            else:
                new_h = max_size[1]
                new_w = int(w * (max_size[1] / h))

            thumbnail = cv2.resize(
                image, (new_w, new_h),
                interpolation=cv2.INTER_AREA
            )

            return thumbnail

        except Exception as e:
            logger.error(f"Error creating thumbnail: {str(e)}")
            return image

    def process_pipeline(self, input_path: str, output_dir: str) -> dict:
        """Полный пайплайн обработки изображения"""
        try:
            # Загружаем
            original = self.load_image(input_path)
            if original is None:
                return {"success": False, "error": "Failed to load image"}

            # Предобработка
            preprocessed = self.preprocess_image(original)

            # Выравнивание
            deskewed = self.deskew_image(preprocessed)

            # Сохраняем обработанное изображение
            base_name = os.path.basename(input_path)
            processed_path = os.path.join(output_dir, f"processed_{base_name}")
            self.save_processed_image(deskewed, processed_path)

            # Создаем миниатюру
            thumbnail = self.create_thumbnail(deskewed)
            thumbnail_path = os.path.join(output_dir, f"thumb_{base_name}")
            cv2.imwrite(thumbnail_path, thumbnail)

            # Сегментация на работы (если нужно)
            works = self.segment_into_works(deskewed)

            # Выделяем текстовые области
            regions = self.extract_text_regions(deskewed)

            return {
                "success": True,
                "original_path": input_path,
                "processed_path": processed_path,
                "thumbnail_path": thumbnail_path,
                "works_count": len(works),
                "regions_count": len(regions),
                "image_stats": {
                    "original_size": original.shape,
                    "processed_size": deskewed.shape,
                    "thumbnail_size": thumbnail.shape
                }
            }

        except Exception as e:
            logger.error(f"Error in processing pipeline: {str(e)}")
            return {"success": False, "error": str(e)}
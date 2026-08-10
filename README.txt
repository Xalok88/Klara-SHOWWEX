
Klara — Ultra UI (PySide6)
==========================
• Glassmorphism + neon + частицы на фоне.
• Пульсирующая кнопка «Запустить анализ».
• Сегменты (Локальный/Облачный, Вайфу/Сток, Женский/Мужской).
• Трей, сворачивание, тост-уведомления, горячие клавиши (Ctrl+K палитра, Ctrl+L очистка лога).
• Звуки: ./audio/startup.mp3 при старте, ./audio/but.mp3 при кликах (если файлы есть).
• История: ./bin/chat_history.json.
• Зависит от вашего klara.py (ищет рядом, использует KlaraCore при наличии), иначе демо-режим.

Запуск:
  pip install PySide6 PySide6-Addons PySide6-Essentials
  python main.py

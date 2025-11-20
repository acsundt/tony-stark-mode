import pyautogui

GESTURE_ACTIONS = {


    "open_palm": lambda: pyautogui.press(""),   # <- used as a transition gesture
    "fist": lambda: pyautogui.press("volumemute"),
    "pinch": lambda:  pyautogui.click(),
    "peace": lambda: pyautogui.press("volumedown"),
    "thumbs_up": lambda: pyautogui.press("volumeup")
}


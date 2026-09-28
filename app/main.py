from application_manager import ApplicationManager
import os
import sys

if sys.platform.startswith("linux"):
    os.environ["XDG_SESSION_TYPE"] = "x11"
    os.environ["GDK_BACKEND"] = "x11"

if __name__ == "__main__":
    print("Realtime? y/n:")
    realtime = input()
    if realtime == "y":
        print("REALTIME ENABLED")
        application_manager = ApplicationManager(True)
    else:
        print("NO REALTIME")
        application_manager = ApplicationManager(False)
    try:
        # Choose programm mode:

        #application_manager.collect_camera_images()
        application_manager.run_with_pictures(realtime_tracking=False) #real application
        #application_manager.run_test_calibration()
        #application_manager.run_with_evaluation_study_2()
        #application_manager.run_with_evaluation_study_3()
    finally:
        if application_manager.camera:
            application_manager.camera.stop_camera()

import os

if __name__ == "__main__":
    """
    | Install necessary library for python (Windows only)
    """
    # Library for numerical computation with image
    os.system("pip install numpy")
    os.system("pip install tensorflow")

    # Library for image processing
    os.system("pip install Pillow")

    # Library for plotting image
    os.system("pip install matplotlib")

    # Library for face recognition
    os.system("pip install dlib")

    # Library for download file
    os.system("pip install wget")

    # Library for make web app
    os.system("pip install streamlit")

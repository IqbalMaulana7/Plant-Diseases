import streamlit as st
import tensorflow as tf
import numpy as np


#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

#Main Page
if(app_mode=="Home"):
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "pertanian.jpg"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Plant Disease Recognition System merupakan sistem yang dapat membantu mengindentifikasi penyakit pada tanaman secara efisien. Cara kerja sistem ini yaitu dengan mengupload gambar atau image dari tanaman yang terkena penyakit, dan sistem akan menganalisisnya untuk mendeteksi penyakit tersebut.

    ### Cara Kerja Sistem :
    1. **Upload Gambar :** Buka halaman **Disease Recognition** dan upload gambar tanaman yang diduga terkena penyakit.
    2. **Analisis :** Sistem akan memproses gambar menggunakan algoritma yang telah dibuat untuk mengidentifikasi potensi penyakit.
    3. **Hasil :** Sistem akan memberikan hasil dan rekomendasi tindakan lebih lanjut terhadap tanaman. 

    ### Kelebihan Sistem Kami :
    - **Akurasi:** Sistem kami menggunakan teknik machine learning untuk mendeteksi penyakit secara akurat.
    - **Ramah Pengguna:** Interface yang sederhana dan intuitif untuk pengalaman pengguna yang lancar.
    - **Cepat dan Efisien:** Memberikan hasil dalam hitungan detik, memungkinkan pengambilan keputusan dengan cepat.

    ### Get Started
    Klik halaman **Disease Recognition** pada sidebar untuk mengupload gambar dan memberikan pengalaman pada sistem kami yaitu Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

#About Project
elif(app_mode=="About"):
    st.header("About Us")
    image_path = "about us.jpg"
    st.image(image_path,use_column_width=True)
    st.markdown("""
                #### About Dataset
                Dataset ini dibuat ulang menggunakan offline augmentation dari dataset asli.
                Dataset ini terdiri dari sekitar 87K rgb gambar daun tanaman yang sehat dan terjerat penyakit yang dikategorikan ke dalam 38 class berbeda. Total dataset dibagi menjadi rasio 80/20 dari set training dan validasi yang menjaga struktur direktori.
                Direktori baru yang berisi 33 gambar uji dibuat untuk tujuan prediksi tanaman.
                #### Content
                1. Train (70295 gambar)
                2. Test (33 gambar)
                3. Validation (17572 gambar)
                """)

#Prediction Page
elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        st.balloons()
        st.write("Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
        # Define disease names and treatments
        disease_names = [
            "Apple Scab", "Apple Black Rot", "Cedar Apple Rust", "Healthy Apple",
            "Healthy Blueberry", "Powdery Mildew on Cherry", "Healthy Cherry",
            "Cercospora Leaf Spot on Corn", "Common Rust on Corn", "Northern Leaf Blight on Corn", "Healthy Corn",
            "Black Rot on Grape", "Esca (Black Measles) on Grape", "Leaf Blight on Grape", "Healthy Grape",
            "Haunglongbing (Citrus Greening) on Orange", "Bacterial Spot on Peach", "Healthy Peach",
            "Bacterial Spot on Bell Pepper", "Healthy Bell Pepper",
            "Early Blight on Potato", "Late Blight on Potato", "Healthy Potato",
            "Healthy Raspberry", "Healthy Soybean", "Powdery Mildew on Squash",
            "Leaf Scorch on Strawberry", "Healthy Strawberry",
            "Bacterial Spot on Tomato", "Early Blight on Tomato", "Late Blight on Tomato",
            "Leaf Mold on Tomato", "Septoria Leaf Spot on Tomato", "Two-spotted Spider Mite on Tomato",
            "Target Spot on Tomato", "Tomato Yellow Leaf Curl Virus", "Tomato Mosaic Virus",
            "Healthy Tomato"
        ]
        treatments = [
            "Treat with fungicides and remove infected leaves.",
            "Prune infected areas and treat with fungicides.",
            "Remove infected leaves and treat with fungicides.",
            "No treatment required, maintain good plant health.",
            "Ensure proper watering and nutrition.",
            "Treat with fungicides and prune infected areas.",
            "No treatment required, maintain good plant health.",
            "Apply fungicides and practice crop rotation.",
            "Apply fungicides and practice crop rotation.",
            "Apply fungicides and practice crop rotation.",
            "No treatment required, maintain good plant health.",
            "Prune and remove infected areas, apply fungicides.",
            "Prune and remove infected areas, apply fungicides.",
            "Prune and remove infected areas, apply fungicides.",
            "No treatment required, maintain good plant health.",
            "Control vector insects and remove infected trees.",
            "Prune and remove infected areas, apply bactericides.",
            "No treatment required, maintain good plant health.",
            "Apply bactericides and practice crop rotation.",
            "No treatment required, maintain good plant health.",
            "Apply fungicides and practice crop rotation.",
            "Apply fungicides and practice crop rotation.",
            "No treatment required, maintain good plant health.",
            "No treatment required, maintain good plant health.",
            "Apply fungicides and practice crop rotation.",
            "Prune and remove infected areas, apply fungicides.",
            "No treatment required, maintain good plant health.",
            "Prune and remove infected areas, apply fungicides.",
            "Prune and remove infected areas, apply fungicides.",
            "Prune and remove infected areas, apply fungicides.",
            "Prune and remove infected areas, apply fungicides.",
            "Prune and remove infected areas, apply fungicides.",
            "Control pests and apply fungicides.",
            "Prune and remove infected areas, apply fungicides.",
            "Control insects and apply pesticides.",
            "No treatment required, maintain good plant health."
        ]
        st.success("Model Memprediksi Adalah Sebuah : {}".format(class_name[result_index]))
        st.write("Nama Penyakit :", disease_names[result_index])
        st.write("Cara Penanganan :", treatments[result_index])
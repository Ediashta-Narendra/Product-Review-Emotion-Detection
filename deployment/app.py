
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tensorflow as tf
from wordcloud import WordCloud
import io


st.write('Tokopedia Product Review Datset')

st.markdown('#### Data Privacy')
st.caption("""
You are entering a restricted environment containing highly sensitive and confidential information owned by Tokopedia. Access is strictly limited to authorized personnel only. Unauthorized access, sharing, or disclosure of this information is strictly prohibited and may lead to 
           disciplinary action and/or legal consequences under applicable regulations. 
           Ensure you are properly authorized before proceeding. Please check the box and press the button below to proceed.
""")


left_column, right_column = st.columns(2)
pressed = left_column.button('Confirm')
if pressed:
  #checkbox
        agree = st.checkbox('I fully understand my authority and my limitations, as well as the companys terms and local laws.')
        if agree:
            st.write('Great!')

'''
# Customer Emotion on Product Review
This page is aimed at exploring the understanding of consumer emotions, 
particularly within the context of product reviews on Tokopedia. 
By analyzing consumer emotions expressed in reviews, 
businesses can gain valuable insights into customer satisfaction and enhance decision-making around product and service improvements.

Emotion detection is essential for sustaining customer loyalty and attracting potential users. 
Prevalence of negative emotions may lead to increased customer churn and deter new users from engaging with the platform. 
Hence, recognizing and addressing consumer emotions is a key strategy in maintaining satisfaction.
'''
#____________________________________________________________________________________________________________

df_ori = pd.read_csv("PRDECT-ID Dataset.csv")
df = df_ori.copy()
#get Customer Review and Emotion Columns Only
df = df[['Customer Review','Emotion']]
df.head(5)
#____________________________________________________________________________________________________________

# Sample DataFrame (replace this with your actual DataFrame)
data = df

# Title of the Streamlit app
st.title("Emotion Distribution Visualization")

# Create emotions variable
emo_counts = df['Emotion'].value_counts().sort_index()

# Define a color palette
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Create a figure with subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))

# Bar plot
bars = axs[0].bar(emo_counts.index.astype(str), emo_counts.values, color=colors)
axs[0].set_title('Bar Plot of Emotion Counts')
axs[0].set_xlabel('')
axs[0].set_ylabel('Number of Reviews')

# Add amounts on top of bar
for bar in bars:
    yval = bar.get_height()
    axs[0].text(bar.get_x() + bar.get_width()/2, yval, str(int(yval)), ha='center', va='bottom')

# Pie chart
axs[1].pie(emo_counts, labels=emo_counts.index.astype(str), autopct='%1.1f%%', startangle=140, colors=colors)
axs[1].set_title('Distribution of Emotions')
axs[1].axis('equal')  # Equal aspect ratio ensures that pie chart is a circle.

# Display the plot in Streamlit
st.pyplot(fig)
st.caption("""figure 1. Distribution of emotions in gathered data""")
# ____________________________________________________________________________________________________________
'''Based on the framework by Shaver et al., in analyzing customers' preoduct review we can classifies emotions into 
five categories—love, happiness, anger, fear, and sadness
—where happiness and love signify positive emotions, while anger, fear, and sadness indicate negative sentiment.
From 5400 gathered product review, we can observe that the percentage of Happy emotion in our data is 33% 
but negative sentiments like anger, fear, and sadness also have comparable percentage.'''
# ____________________________________________________________________________________________________________

# Title of the Streamlit app
st.title("Random Emotion Samples")
'''Here, you can observe the gathered sample for each emotions we've been categorized. We may tell an intrgiguing insight : 
negative sentiment may poseess longer token/text than the possitive one, just click each of category button to prove that statement :)'''

st.caption('''Re-clik for generate other randomized sample''')
# Create a button for each Emotion
if st.button('Show Anger Samples'):
    sample_anger = df[df['Emotion'] == 'Anger'].sample(n=3)
    st.write('### SAMPLE OF ANGER EMOTIONS')
    st.write(sample_anger.values)

if st.button('Show Fear Samples'):
    sample_fear = df[df['Emotion'] == 'Fear'].sample(n=3)
    st.write('### SAMPLE OF FEAR EMOTIONS')
    st.write(sample_fear.values)

if st.button('Show Happy Samples'):
    sample_happy = df[df['Emotion'] == 'Happy'].sample(n=3)
    st.write('### SAMPLE OF HAPPY EMOTIONS')
    st.write(sample_happy.values)

if st.button('Show Love Samples'):
    sample_love = df[df['Emotion'] == 'Love'].sample(n=3)
    st.write('### SAMPLE OF LOVE EMOTIONS')
    st.write(sample_love.values)

if st.button('Show Sadness Samples'):
    sample_sadness = df[df['Emotion'] == 'Sadness'].sample(n=3)
    st.write('### SAMPLE OF SADNESS EMOTIONS')
    st.write(sample_sadness.values)

#____________________________________________________________________________________________________________

# Load the model and vectorizer
@st.cache_resource  # Caches the model and vectorizer for faster loading
def load_resources():
    model = tf.keras.models.load_model('emotion_detection_model.keras')
    vectorizer = joblib.load('vectorizer.joblib')
    return model, vectorizer

model, vectorizer = load_resources()

# Define the class labels
label_mapping = {0: 'Anger', 1: 'Fear', 2: 'Happy', 3: 'Love', 4: 'Sadness'}

# Define the prediction function
def predict_emotion(text):
    # Step 1: Vectorize the input text
    text_vectorized = vectorizer.transform([text]).toarray()
    
    # Step 2: Make a prediction
    predictions = model.predict(text_vectorized)
    
    # Step 3: Convert probabilities to percentages and associate with class names
    prediction_percentages = (predictions[0] * 100).round(2)
    class_probabilities = {label_mapping[i]: f"{prob:.2f}%" for i, prob in enumerate(prediction_percentages)}
    
    # Step 4: Find the emotion with the highest probability
    predicted_label = np.argmax(predictions, axis=1)[0]
    predicted_emotion = label_mapping[predicted_label]
    
    return predicted_emotion, class_probabilities

# Streamlit UI
st.title("Emotion Prediction on Product Review")
st.write("For quick detection pleae paste the customer review here, we have build an ANN model to predict the emotion felt when customer write the review. We can ensure the accuracy will be enough, since we have do trial-error in our model building, at least the prediction resulting more than 50% accuracy.")

# Text input for user to enter their text
input_text = st.text_area("Enter a product review to predict its emotion:")

# Predict button
if st.button("Predict Emotion"):
    if input_text:
        # Run the prediction
        predicted_emotion, class_probabilities = predict_emotion(input_text)
        
        # Display the results
        st.write("### Input Document:")
        st.write(input_text)
        st.write("### Predicted Emotion:")
        st.write(predicted_emotion)
        st.write("### Prediction Probabilities by Class:")
        st.write(class_probabilities)
    else:
        st.warning("Please enter some text to predict its emotion.")

''':rage: :fearful: :grin: :heart_eyes: :disappointed_relieved:  '''
#____________________________________________________________________________________________________________
st.title("Product Review in WordCloud")
st.write("To finish up, this the Word Cloud for the raw vocabulary. In ther process we've been preprocess the text to remove the exceed vocab.")
# Define overall words as variable
word_all = df['Customer Review'].values

# Generate WordCloud
WC_all = WordCloud(background_color='white', colormap='viridis', collocations=False, width=2000, height=1200).generate(" ".join(word_all))

# Create a Matplotlib figure
plt.figure(figsize=(15, 10))
plt.axis('off')
plt.title("WordCloud for all Reviews", fontsize=12)
plt.imshow(WC_all, interpolation='bilinear')

# Save the plot to a BytesIO object
buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)

# Display WordCloud in Streamlit
st.image(buf, caption="WordCloud for all Reviews", use_column_width=True)

# Optional: clear the figure to avoid display issues in Streamlit
plt.clf()
#____________________________________________________________________________
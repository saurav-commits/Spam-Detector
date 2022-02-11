import pickle
import streamlit as st

model = pickle.load(open("spam.pkl","rb"))
cv = pickle.load(open("vectorizer.pkl","rb"))

def main():
	st.title("Email Detector")
	st.subheader("Build with streamlit and python")
	msg = st.text_input("Enter the message:")
	if st.button("Predict"):
		data = [msg]
		vect = cv.transform(data).toarray()
		prediction = model.predict(vect)
		result = prediction[0]
		if result == 1:
			st.error("This message is a spam mail")
		else:
			st.success("This message is a ham mail")

main()
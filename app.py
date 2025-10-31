# app.py
import streamlit as st
import joblib, os, numpy as np

st.set_page_config(page_title='ADUSTECH Antimicrobial Predictor App', layout='centered')
st.title('🧬 ADUSTECH Antimicrobial Predictor App')
st.markdown('**Contributors:** Mujahid Nura, Abdulwahid Isah Adamu, **Sefiya Isah**  \n\n**Purpose:** Predict resistance to Ciprofloxacin & Augmentin and recommend alternatives (rule-based).')

st.write('---')

st.subheader('Enter inhibition zone diameters (mm)')
col1, col2 = st.columns(2)
with col1:
    imipenem = st.number_input('Imipenem (mm)', min_value=0.0, max_value=100.0, value=25.0, step=1.0, format='%.1f')
    ceftazidime = st.number_input('Ceftazidime (mm)', min_value=0.0, max_value=100.0, value=20.0, step=1.0, format='%.1f')
with col2:
    gentamicin = st.number_input('Gentamicin (mm)', min_value=0.0, max_value=100.0, value=20.0, step=1.0, format='%.1f')
    organism = st.selectbox('Organism (note: model currently trained WITHOUT organism labels)', options=['Escherichia coli','Staphylococcus aureus','Klebsiella pneumoniae'])

st.write('---')

MODEL_PATH = 'models/amr_model.pkl'
if not os.path.exists(MODEL_PATH):
    st.error(f'Model file not found at {MODEL_PATH}. Please ensure amr_model.pkl is in the models/ folder.')
else:
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        st.error('Failed to load model. Ensure the model was saved with scikit-learn==1.5.1 and joblib==1.4.x.')
        st.exception(e)
        model = None

def recommend(sample, predicted):
    # sample: dict with IMIPENEM, CEFTAZIDIME, GENTAMICIN
    ab = [('Imipenem', sample.get('IMIPENEM',0)), ('Ceftazidime', sample.get('CEFTAZIDIME',0)), ('Gentamicin', sample.get('GENTAMICIN',0))]
    ab_sorted = sorted(ab, key=lambda x: -x[1])
    rec = {}
    if int(predicted[0])==1:
        rec['Ciprofloxacin_alternatives'] = [n for n,v in ab_sorted][:2]
    if int(predicted[1])==1:
        rec['Augmentin_alternatives'] = [n for n,v in ab_sorted][:2]
    return rec

if st.button('🔍 Predict'):
    if 'model' not in globals() or model is None:
        st.error('Model not available. Please upload the trained model (models/amr_model.pkl).')
    else:
        X = np.array([[imipenem, ceftazidime, gentamicin]])
        pred = model.predict(X)[0]
        cip_text = 'Resistant' if int(pred[0])==1 else 'Susceptible'
        aug_text = 'Resistant' if int(pred[1])==1 else 'Susceptible'
        if pred[0]==1:
            st.error(f'Ciprofloxacin: **{cip_text}**')
        else:
            st.success(f'Ciprofloxacin: **{cip_text}**')
        if pred[1]==1:
            st.error(f'Augmentin: **{aug_text}**')
        else:
            st.success(f'Augmentin: **{aug_text}**')
        sample = {'IMIPENEM': imipenem, 'CEFTAZIDIME': ceftazidime, 'GENTAMICIN': gentamicin}
        recs = recommend(sample, pred)
        if recs:
            st.markdown('### 🩺 Recommendations (rule-based)')
            for k,v in recs.items():
                st.warning(f"{k}: {', '.join(v)}")
        st.info('Note: Organism selector is for UX. The current model was trained without organism labels. Provide organism-labelled data to retrain.')

from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import cv2
import pickle
import base64
import io
import os

app = Flask(__name__)

# ── Treatment Recommendations ─────────────────────────────────
TREATMENTS = {
    "Apple___Apple_scab": [
        "Remove and destroy all infected leaves and fruit immediately — do not leave them on the ground.",
        "Spray your apple tree with a fungicide containing captan or myclobutanil every 7 days during wet weather.",
        "Prune branches to open up the tree canopy so air can flow through and leaves dry faster after rain.",
        "Next season, apply a protective fungicide spray starting at bud break before symptoms appear."
    ],
    "Apple___Black_rot": [
        "Cut off and dispose of all infected branches at least 8 inches below the visible infection — burn or bag them.",
        "Remove any mummified fruits still hanging on the tree as they spread the disease to healthy fruit.",
        "Apply a copper-based fungicide spray every 10 days from pink bud stage through harvest.",
        "Improve drainage around your trees and avoid overhead irrigation to keep leaves dry."
    ],
    "Apple___Cedar_apple_rust": [
        "Remove any nearby cedar or juniper trees within 100 metres if possible — they host this disease.",
        "Apply a fungicide containing myclobutanil or propiconazole starting at pink bud and every 7-10 days after.",
        "Choose rust-resistant apple varieties for future planting such as Liberty or Enterprise.",
        "Consult your local agricultural extension office if the infection is severe or spreading rapidly."
    ],
    "Apple___healthy": [
        "Your apple plant looks healthy — no treatment needed at this time.",
        "Continue regular monitoring every 1-2 weeks especially during wet weather.",
        "Maintain good pruning practices to keep air circulating through the canopy.",
        "Apply a preventive fungicide spray at the start of the growing season as a precaution."
    ],
    "Blueberry___healthy": [
        "Your blueberry plant looks healthy — no treatment needed at this time.",
        "Make sure soil pH stays between 4.5 and 5.5 — blueberries need acidic soil to thrive.",
        "Water consistently at the base of the plant and add mulch to retain moisture.",
        "Monitor regularly for any early signs of disease especially during humid weather."
    ],
    "Cherry_(including_sour)___Powdery_mildew": [
        "Remove and destroy all infected leaves, shoots, and fruit showing white powdery coating immediately.",
        "Spray with a sulfur-based or potassium bicarbonate fungicide every 7-10 days until symptoms clear.",
        "Avoid overhead watering — water at the base only and do it in the morning so plants dry before evening.",
        "Prune to improve air circulation and reduce humidity around the plant."
    ],
    "Cherry_(including_sour)___healthy": [
        "Your cherry plant looks healthy — no treatment needed at this time.",
        "Continue watering at the base and avoid wetting the leaves unnecessarily.",
        "Monitor regularly for early signs of powdery mildew or leaf spot.",
        "Apply a balanced fertilizer in early spring to support healthy growth."
    ],
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": [
        "Remove and destroy severely infected leaves to reduce the source of spores spreading.",
        "Apply a strobilurin-based fungicide such as azoxystrobin at first sign of symptoms.",
        "Rotate your crops — do not plant corn in the same field for at least two seasons.",
        "Choose gray leaf spot resistant corn hybrids for your next planting season."
    ],
    "Corn_(maize)___Common_rust_": [
        "Apply a fungicide containing propiconazole or azoxystrobin as soon as you see orange pustules appearing.",
        "Plant rust-resistant corn hybrids in future seasons to reduce the risk significantly.",
        "Monitor your crop every few days during warm humid weather when rust spreads fastest.",
        "If less than 25 percent of leaves are affected, the crop may still recover without treatment."
    ],
    "Corn_(maize)___Northern_Leaf_Blight": [
        "Apply a fungicide containing propiconazole at the first sign of tan lesions on the leaves.",
        "Rotate crops and avoid planting corn in the same field more than two years in a row.",
        "Till crop residue after harvest to reduce the amount of disease carrying material in the soil.",
        "Use resistant hybrid varieties in future planting to significantly reduce risk."
    ],
    "Corn_(maize)___healthy": [
        "Your corn plant looks healthy — no treatment needed at this time.",
        "Continue monitoring every few days especially during warm and humid conditions.",
        "Ensure adequate nitrogen fertilization to keep plants vigorous and more resistant to disease.",
        "Scout regularly for early signs of rust or blight so you can act quickly."
    ],
    "Grape___Black_rot": [
        "Remove all infected berries, leaves, and mummified fruit immediately and dispose of them far from the vineyard.",
        "Apply a fungicide containing mancozeb or myclobutanil every 7-10 days starting at bud break.",
        "Prune vines to improve air circulation and reduce the humidity that this disease thrives in.",
        "Avoid wetting the foliage when irrigating — use drip irrigation at the base of the vines."
    ],
    "Grape___Esca_(Black_Measles)": [
        "There is no cure for Esca — remove and destroy any severely infected vines to prevent spread.",
        "Avoid large pruning wounds which are the main entry point for this disease.",
        "Protect fresh pruning cuts immediately with a wound sealant or fungicide paste.",
        "Consult an agricultural expert — Esca is serious and management requires a long-term strategy."
    ],
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": [
        "Remove and destroy infected leaves immediately to reduce the spread of spores.",
        "Apply a copper-based fungicide spray every 10 days during the growing season.",
        "Improve air circulation by pruning and trellising vines properly.",
        "Avoid overhead irrigation — keep foliage as dry as possible."
    ],
    "Grape___healthy": [
        "Your grape vine looks healthy — no treatment needed at this time.",
        "Continue regular monitoring especially after rain when fungal diseases spread fastest.",
        "Maintain proper pruning and trellis management for good air circulation.",
        "Apply a preventive copper spray at bud break as a standard precaution."
    ],
    "Orange___Haunglongbing_(Citrus_greening)": [
        "There is no cure for citrus greening — infected trees must be removed and destroyed immediately.",
        "Control the Asian citrus psyllid insect with insecticide sprays as it is the carrier of this disease.",
        "Do not move infected plant material to other areas — this disease spreads very easily.",
        "Contact your local agricultural authority immediately — citrus greening is a notifiable disease in most regions."
    ],
    "Peach___Bacterial_spot": [
        "Apply a copper-based bactericide spray every 7-10 days starting at bud break and continuing through the season.",
        "Avoid overhead irrigation — water at the base only and in the morning so leaves dry quickly.",
        "Remove and destroy severely infected shoots and leaves to reduce the source of bacteria.",
        "Choose bacterial spot resistant peach varieties for future planting."
    ],
    "Peach___healthy": [
        "Your peach plant looks healthy — no treatment needed at this time.",
        "Apply a preventive copper spray at bud swell each spring as a standard precaution.",
        "Monitor regularly for signs of bacterial spot especially after rain and wind events.",
        "Ensure good drainage and avoid water stress which weakens trees and makes them more susceptible."
    ],
    "Pepper,_bell___Bacterial_spot": [
        "Remove and destroy all infected leaves and fruit — do not compost them.",
        "Apply a copper-based bactericide spray every 5-7 days during wet weather.",
        "Avoid working with plants when they are wet as this spreads bacteria from plant to plant.",
        "Use certified disease-free seeds and transplants for your next crop."
    ],
    "Pepper,_bell___healthy": [
        "Your bell pepper plant looks healthy — no treatment needed at this time.",
        "Water at the base of the plant and avoid wetting the foliage.",
        "Monitor regularly for early signs of bacterial spot especially after rain.",
        "Rotate pepper crops every 2-3 years to prevent soil-borne disease buildup."
    ],
    "Potato___Early_blight": [
        "Remove lower infected leaves immediately and dispose of them away from the field.",
        "Apply a fungicide containing chlorothalonil or mancozeb every 7-10 days during wet weather.",
        "Avoid overhead irrigation — water at the base in the morning so plants dry before evening.",
        "Add mulch around plants to prevent soil splash which spreads the fungus to lower leaves."
    ],
    "Potato___Late_blight": [
        "Act immediately — late blight spreads extremely fast and can destroy an entire crop within days.",
        "Apply a fungicide containing chlorothalonil or metalaxyl right away and repeat every 5-7 days.",
        "Remove and destroy all infected plant material — do not leave it in the field or compost it.",
        "Harvest any healthy tubers as soon as possible if more than 30 percent of the foliage is affected."
    ],
    "Potato___healthy": [
        "Your potato plant looks healthy — no treatment needed at this time.",
        "Scout regularly for early blight and late blight especially during cool wet weather.",
        "Apply a preventive fungicide spray when conditions are favorable for disease.",
        "Hill up soil around plants to protect developing tubers from light and disease."
    ],
    "Raspberry___healthy": [
        "Your raspberry plant looks healthy — no treatment needed at this time.",
        "Prune out old fruited canes after harvest to improve air circulation and reduce disease risk.",
        "Monitor for signs of cane blight, anthracnose, and gray mold.",
        "Apply a balanced fertilizer in early spring to support strong healthy growth."
    ],
    "Soybean___healthy": [
        "Your soybean plant looks healthy — no treatment needed at this time.",
        "Scout regularly for signs of sudden death syndrome, frogeye leaf spot, and soybean rust.",
        "Maintain proper plant spacing to allow good air circulation through the canopy.",
        "Rotate with non-legume crops every season to reduce soil-borne disease pressure."
    ],
    "Squash___Powdery_mildew": [
        "Remove heavily infected leaves immediately to slow the spread of the white powdery fungus.",
        "Spray with potassium bicarbonate or a neem oil solution every 7 days until symptoms clear.",
        "Water at the base of the plant only — wet foliage encourages this disease to spread.",
        "Plant squash in full sun with adequate spacing so air can circulate between plants."
    ],
    "Strawberry___Leaf_scorch": [
        "Remove and destroy all infected leaves showing dark purple spots with light centers.",
        "Apply a fungicide containing captan or myclobutanil every 10-14 days during wet conditions.",
        "Avoid overhead irrigation and water in the morning so leaves dry before nightfall.",
        "Renovate strawberry beds after harvest by removing old leaves and applying fresh mulch."
    ],
    "Strawberry___healthy": [
        "Your strawberry plant looks healthy — no treatment needed at this time.",
        "Remove runners regularly to direct energy into fruit production.",
        "Monitor for signs of leaf scorch, gray mold, and verticillium wilt.",
        "Apply a balanced fertilizer after harvest to support next season's growth."
    ],
    "Tomato___Bacterial_spot": [
        "Remove and destroy all infected leaves and fruit immediately.",
        "Apply a copper-based bactericide spray every 5-7 days especially during warm wet weather.",
        "Avoid working with wet plants and always sanitize tools between uses.",
        "Use certified disease-free transplants and rotate tomato crops every 2-3 years."
    ],
    "Tomato___Early_blight": [
        "Remove all infected lower leaves immediately and dispose of them away from the garden.",
        "Apply chlorothalonil or copper fungicide every 7-10 days during wet weather.",
        "Mulch around the base of the plant to stop soil from splashing onto lower leaves.",
        "Water at the base only and avoid wetting the foliage — water in the morning."
    ],
    "Tomato___Late_blight": [
        "Act immediately — late blight spreads rapidly and can destroy plants within days.",
        "Apply a fungicide containing chlorothalonil or metalaxyl right away and every 5-7 days.",
        "Remove and destroy all infected plant material — bag it and throw it away.",
        "Harvest any ripe or near-ripe tomatoes now before the disease reaches them."
    ],
    "Tomato___Leaf_Mold": [
        "Improve greenhouse or tunnel ventilation immediately to reduce humidity around plants.",
        "Remove and destroy all leaves showing yellow patches with olive-green mold underneath.",
        "Apply a fungicide containing chlorothalonil every 7-10 days until conditions improve.",
        "Avoid overhead watering and ensure adequate plant spacing for air circulation."
    ],
    "Tomato___Septoria_leaf_spot": [
        "Remove infected lower leaves immediately showing small circular spots with dark borders.",
        "Apply mancozeb or chlorothalonil fungicide every 7-10 days during wet weather.",
        "Mulch around plants to prevent soil splash from spreading fungal spores to leaves.",
        "Rotate tomatoes with non-related crops and avoid planting in the same spot two years in a row."
    ],
    "Tomato___Spider_mites Two-spotted_spider_mite": [
        "Spray plants thoroughly with water to physically knock mites off the leaves — do this in the morning.",
        "Apply neem oil or insecticidal soap spray every 5-7 days covering both sides of leaves.",
        "Increase humidity around plants — spider mites thrive in hot dry conditions.",
        "Consult an agricultural expert if the infestation is severe as miticide may be needed."
    ],
    "Tomato___Target_Spot": [
        "Remove and destroy all infected leaves showing concentric ring patterns immediately.",
        "Apply a fungicide containing chlorothalonil or azoxystrobin every 7-10 days.",
        "Improve air circulation by staking and pruning plants to reduce leaf wetness.",
        "Avoid overhead irrigation and water early in the day so foliage dries before evening."
    ],
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": [
        "There is no cure — remove and destroy infected plants immediately to prevent spread.",
        "Control whiteflies with yellow sticky traps and insecticide sprays as they carry this virus.",
        "Cover young transplants with fine insect netting to prevent whitefly access.",
        "Plant virus-resistant tomato varieties in future seasons."
    ],
    "Tomato___Tomato_mosaic_virus": [
        "There is no cure — remove and destroy infected plants immediately.",
        "Wash hands thoroughly and disinfect all tools before touching other plants.",
        "Control aphids which spread this virus using insecticidal soap or neem oil.",
        "Use virus-resistant tomato varieties and certified disease-free seeds for future crops."
    ],
    "Tomato___healthy": [
        "Your tomato plant looks healthy — no treatment needed at this time.",
        "Continue watering at the base and stake plants to keep foliage off the ground.",
        "Scout weekly for early signs of blight, leaf spot, and pest damage.",
        "Apply a balanced fertilizer every 2-3 weeks to support healthy fruit production."
    ]
}

def get_treatment(disease_class):
    return TREATMENTS.get(disease_class, [
        "Remove and destroy all visibly infected leaves immediately.",
        "Apply a copper-based fungicide every 7-10 days until symptoms clear.",
        "Water at the base of the plant only and avoid overhead irrigation.",
        "Consult a local agricultural expert if symptoms worsen or spread rapidly."
    ])

# ── Load Model ────────────────────────────────────────────────
def load_model():
    base_model = tf.keras.applications.MobileNetV2(
        weights=None, include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    model = tf.keras.models.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(38, activation='softmax')
    ])

    model.build((None, 224, 224, 3))

    with open('model_weights.pkl', 'rb') as f:
        weights = pickle.load(f)
    model.set_weights(weights)
    model(tf.zeros((1, 224, 224, 3)))

    with open('class_names.json') as f:
        class_names = json.load(f)

    return model, class_names

model, class_names = load_model()

# ── Grad-CAM ───────────────────────────────────────
def make_gradcam(img_tensor, model):
    try:
        last_conv_layer = None
        for layer in model.layers[0].layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer

        grad_model = tf.keras.models.Model(
            inputs=model.layers[0].input,
            outputs=[last_conv_layer.output, model.layers[0].output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, base_outputs = grad_model(img_tensor)
            full_output = model.layers[1](base_outputs)
            full_output = model.layers[2](full_output)
            full_output = model.layers[3](full_output)
            top_class = tf.argmax(full_output[0])
            loss = full_output[:, top_class]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()

        img = img_tensor[0].numpy()
        img = (img * 255).astype(np.uint8)
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap_colored = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)

        return overlay
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        return None

# ── Routes ────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img = Image.open(file.stream).convert('RGB')

    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_tensor = tf.cast(np.expand_dims(img_array, axis=0), tf.float32)

    predictions = model.predict(img_tensor)
    gradcam_img = make_gradcam(img_tensor, model)

    top3_idx = np.argsort(predictions[0])[::-1][:3]
    top_confidence = float(predictions[0][top3_idx[0]] * 100)
    top_class = class_names[top3_idx[0]]
    top_name = top_class.replace('___', ' — ').replace('_', ' ')

    preds = []
    for idx in top3_idx:
        preds.append({
            'name': class_names[idx].replace('___', ' — ').replace('_', ' '),
            'confidence': round(float(predictions[0][idx] * 100), 1)
        })

    treatment = get_treatment(top_class)

    gradcam_b64 = None
    if gradcam_img is not None:
        pil_img = Image.fromarray(gradcam_img.astype(np.uint8))
        buffer = io.BytesIO()
        pil_img.save(buffer, format='JPEG', quality=85)
        buffer.seek(0)
        gradcam_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return jsonify({
        'disease': top_name,
        'confidence': round(top_confidence, 1),
        'predictions': preds,
        'treatment': treatment,
        'gradcam': gradcam_b64
    })

if __name__ == '__main__':
    app.run(debug=True)
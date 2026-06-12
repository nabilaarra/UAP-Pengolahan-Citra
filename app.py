import base64
import io
import numpy as np
import pandas as pd
import cv2
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from skimage.feature import graycomatrix, graycoprops
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

FEATURE_COLUMNS = ["Contrast", "Correlation", "Energy", "Homogeneity"]

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

train_df = pd.read_csv('hasil_train.csv')
test_df = pd.read_csv('hasil_test.csv')

LABELS = ['ripe', 'unripe']


def build_knn(k):
    model = KNeighborsClassifier(n_neighbors=max(1, int(k)))
    model.fit(train_df[FEATURE_COLUMNS], train_df['Label'])
    return model


def extract_glcm_features(gray_image):
    if gray_image.ndim == 3:
        gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(
        gray_image,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )
    return {
        'Contrast': float(graycoprops(glcm, 'contrast')[0, 0]),
        'Correlation': float(graycoprops(glcm, 'correlation')[0, 0]),
        'Energy': float(graycoprops(glcm, 'energy')[0, 0]),
        'Homogeneity': float(graycoprops(glcm, 'homogeneity')[0, 0])
    }


def make_prediction(features, k):
    model = build_knn(k)
    values = np.array([[
        float(features['Contrast']),
        float(features['Correlation']),
        float(features['Energy']),
        float(features['Homogeneity'])
    ]])
    label = model.predict(values)[0]
    distances, neighbors = model.kneighbors(values, n_neighbors=min(len(train_df), max(1, int(k))))
    neighbor_info = []
    for dist, idx in zip(distances[0], neighbors[0]):
        neighbor_row = train_df.iloc[idx]
        neighbor_info.append({
            'index': int(idx + 1),
            'label': str(neighbor_row['Label']),
            'distance': float(dist),
            'features': {
                'Contrast': float(neighbor_row['Contrast']),
                'Correlation': float(neighbor_row['Correlation']),
                'Energy': float(neighbor_row['Energy']),
                'Homogeneity': float(neighbor_row['Homogeneity'])
            }
        })
    confidence = 1.0 - float(np.mean(distances[0]) / (np.max(distances[0]) + 1e-6))
    return {
        'label': label,
        'confidence': max(0.0, min(1.0, confidence)),
        'features': {
            'Contrast': float(features['Contrast']),
            'Correlation': float(features['Correlation']),
            'Energy': float(features['Energy']),
            'Homogeneity': float(features['Homogeneity'])
        },
        'nearest': neighbor_info
    }


def evaluation_for_k(k):
    model = build_knn(k)
    y_pred = model.predict(test_df[FEATURE_COLUMNS])
    cm = confusion_matrix(test_df['Label'], y_pred, labels=LABELS)
    acc = accuracy_score(test_df['Label'], y_pred)
    tp = int(cm[0, 0])
    fn = int(cm[0, 1])
    fp = int(cm[1, 0])
    tn = int(cm[1, 1])
    precision_ripe = tp / (tp + fp) if (tp + fp) else 0.0
    recall_ripe = tp / (tp + fn) if (tp + fn) else 0.0
    precision_unripe = tn / (tn + fn) if (tn + fn) else 0.0
    recall_unripe = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        'k': int(k),
        'accuracy': float(acc * 100),
        'confusion_matrix': cm.tolist(),
        'labels': LABELS,
        'metrics': {
            'accuracy': float(acc * 100),
            'precision_ripe': float(precision_ripe * 100),
            'recall_ripe': float(recall_ripe * 100),
            'precision_unripe': float(precision_unripe * 100),
            'recall_unripe': float(recall_unripe * 100)
        },
        'feature_summary': {
            label: {
                feature: float(value)
                for feature, value in train_df[train_df['Label'] == label][FEATURE_COLUMNS].mean().items()
            }
            for label in LABELS
        }
    }


def dataset_summary():
    training_counts = train_df['Label'].value_counts().to_dict()
    testing_counts = test_df['Label'].value_counts().to_dict()
    return {
        'total_train': int(train_df.shape[0]),
        'total_test': int(test_df.shape[0]),
        'train_distribution': {
            'ripe': int(training_counts.get('ripe', 0)),
            'unripe': int(training_counts.get('unripe', 0))
        },
        'test_distribution': {
            'ripe': int(testing_counts.get('ripe', 0)),
            'unripe': int(testing_counts.get('unripe', 0))
        }
    }


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('.', path)


@app.route('/api/dataset')
def api_dataset():
    model = build_knn(5)
    predictions = model.predict(test_df[FEATURE_COLUMNS])
    test_records = []
    for idx, row in test_df.iterrows():
        label = str(row['Label'])
        pred = str(predictions[idx])
        test_records.append({
            'Contrast': float(row['Contrast']),
            'Correlation': float(row['Correlation']),
            'Energy': float(row['Energy']),
            'Homogeneity': float(row['Homogeneity']),
            'Label': label,
            'prediction': pred,
            'status': 'correct' if pred == label else 'wrong'
        })

    return jsonify({
        'summary': dataset_summary(),
        'train': train_df[FEATURE_COLUMNS + ['Label']].to_dict(orient='records'),
        'test': test_records,
        'predictions': evaluation_for_k(5)
    })


@app.route('/api/predict/manual', methods=['POST'])
def api_predict_manual():
    body = request.get_json() or {}
    required = ['Contrast', 'Correlation', 'Energy', 'Homogeneity', 'k']
    if not all(key in body for key in required):
        return jsonify({'error': 'Semua fitur dan parameter K harus diberikan.'}), 400
    try:
        response = make_prediction(body, body['k'])
        return jsonify({'success': True, 'result': response})
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/api/predict/image', methods=['POST'])
def api_predict_image():
    if 'image' not in request.files:
        return jsonify({'error': 'File gambar tidak ditemukan.'}), 400
    file_storage = request.files['image']
    if file_storage.filename == '':
        return jsonify({'error': 'Nama file kosong.'}), 400
    try:
        file_bytes = file_storage.read()
        array = np.frombuffer(file_bytes, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('Gagal membuka gambar. Pastikan format JPG/PNG valid.')
        features = extract_glcm_features(image)
        k = request.form.get('k', 5)
        result = make_prediction(features, k)
        return jsonify({'success': True, 'result': result})
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/api/eval')
def api_eval():
    k = request.args.get('k', 5)
    try:
        k = int(k)
        eval_data = evaluation_for_k(k)
        k_curve = [
            {'k': value, 'accuracy': evaluation_for_k(value)['accuracy']}
            for value in [1, 3, 5, 7, 9, 11]
        ]
        eval_data['k_curve'] = k_curve
        return jsonify({'success': True, 'evaluation': eval_data})
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)

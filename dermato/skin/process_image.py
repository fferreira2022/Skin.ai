import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import operator as op
from collections import Counter


def SvmPredictor_I(image_path):
    data = []
    image = imread(image_path)
    img = resize(image, (224, 224))
    data.append(img.flatten())
    data = np.asarray(data)

    predictions = []
    # accuracy score moyen de SvmPredictor_I = 79.28 %
    with open('skin\\best_estimators_I\\best_estimator_AK_BCC_68.pkl', 'rb') as f:
        classifier1 = pickle.load(f)
    output = classifier1.predict(data)
    if output == 0:
        output = 'actinic keratosis'
    elif output == 1:
        output = 'basal cell carcinoma'
    predictions.append(output)
    
    with open('skin\\best_estimators_I\\best_estimator_AK_BKL_73.pkl', 'rb') as f:
        classifier2 = pickle.load(f)
    output = classifier2.predict(data)
    if output == 0:
        output = 'actinic keratosis'
    elif output == 1:
        output = 'benign keratosis-like lesion'
    predictions.append(output)
    
    with open('skin\\best_estimators_I\\best_estimator_AK_DF_81.pkl', 'rb') as f:
        classifier3 = pickle.load(f)
    output = classifier3.predict(data)
    if output == 0:
        output = 'actinic keratosis'
    elif output == 1:
        output = 'dermatofibroma'
    predictions.append(output)
    
    with open('skin\\best_estimators_I\\best_estimator_AK_MEL_90.pkl', 'rb') as f:
        classifier4 = pickle.load(f)
    output = classifier4.predict(data)
    if output == 0:
        output = 'actinic keratosis'
    elif output == 1:
        output = 'melanoma'
    predictions.append(output)
    
    with open('skin\\best_estimators_I\\best_estimator_AK_NV_87.pkl', 'rb') as f:
        classifier5 = pickle.load(f)
    output = classifier5.predict(data)
    if output == 0:
        output = 'actinic keratosis'
    elif output == 1:
        output = 'nevus'
    predictions.append(output)
    
    with open('skin\\best_estimators_I\\best_estimator_AK_SCC_70.pkl', 'rb') as f:
        classifier6 = pickle.load(f)
    output = classifier6.predict(data)
    if output == 0:
        output = 'actinic keratosis'
    elif output == 1:
        output = 'squamous cell carcinoma'
    predictions.append(output)
    
    with open('skin\\best_estimators_I\\best_estimator_AK_VASC_90.pkl', 'rb') as f:
        classifier7 = pickle.load(f)
    output = classifier7.predict(data)
    if output == 0:
        output = 'actinic keratosis'
    elif output == 1:
        output = 'vascular lesion'
    predictions.append(output)
    
    with open('skin\\best_estimators_I\\best_estimator_BCC_BKL_70.pkl', 'rb') as f:
        classifier8 = pickle.load(f)
    output = classifier8.predict(data)
    if output == 0:
        output = 'basal cell carcinoma'
    elif output == 1:
        output = 'benign keratosis-like lesion'
    predictions.append(output)
    
    with open('skin\\best_estimators_I\\best_estimator_BCC_DF_77.pkl', 'rb') as f:
        classifier9 = pickle.load(f)
    output = classifier9.predict(data)
    if output == 0:
        output = 'basal cell carcinoma'
    elif output == 1:
        output = 'dermatofibroma'
    predictions.append(output)
    
    with open('skin\\best_estimators_I\\best_estimator_BCC_MEL_82.pkl', 'rb') as f:
        classifier10 = pickle.load(f)
    output = classifier10.predict(data)
    if output == 0:
        output = 'basal cell carcinoma'
    elif output == 1:
        output = 'melanoma'
    predictions.append(output)
    
    with open('skin\\best_estimators_I\\best_estimator_BCC_NV_82.pkl', 'rb') as f:
        classifier11 = pickle.load(f)
    output = classifier11.predict(data)
    if output == 0:
        output = 'basal cell carcinoma'
    elif output == 1:
        output = 'nevus'
    predictions.append(output)
    
    with open('skin\\best_estimators_I\\best_estimator_BCC_SCC_72.pkl', 'rb') as f:
        classifier12 = pickle.load(f)
    output = classifier12.predict(data)
    if output == 0:
        output = 'basal cell carcinoma'
    elif output == 1:
        output = 'squamous cell carcinoma'
    predictions.append(output)
    
    with open('skin\\best_estimators_I\\best_estimator_BCC_VASC_83.pkl', 'rb') as f:
        classifier13 = pickle.load(f)
    output = classifier13.predict(data)
    if output == 0:
        output = 'basal cell carcinoma'
    elif output == 1:
        output = 'vascular lesion'
    predictions.append(output)
    
    with open('skin\\best_estimators_I\\best_estimator_BKL_DF_72.pkl', 'rb') as f:
        classifier14 = pickle.load(f)
    output = classifier14.predict(data)
    if output == 0:
        output = 'benign keratosis-like lesion'
    elif output == 1:
        output = 'dermatofibroma'
    predictions.append(output)
    
    with open('skin\\best_estimators_I\\best_estimator_BKL_MEL_71.pkl', 'rb') as f:
        classifier15 = pickle.load(f)
    output = classifier15.predict(data)
    if output == 0:
        output = 'benign keratosis-like lesion'
    elif output == 1:
        output = 'melanoma'
    predictions.append(output)
    
    with open('skin\\best_estimators_I\\best_estimator_BKL_NV_75.pkl', 'rb') as f:
        classifier16 = pickle.load(f)
    output = classifier16.predict(data)
    if output == 0:
        output = 'benign keratosis-like lesion'
    elif output == 1:
        output = 'nevus'
    predictions.append(output)
    
    with open('skin\\best_estimators_I\\best_estimator_BKL_SCC_76.pkl', 'rb') as f:
        classifier17 = pickle.load(f)
    output = classifier17.predict(data)
    if output == 0:
        output = 'benign keratosis-like lesion'
    elif output == 1:
        output = 'squamous cell carcinoma'
    predictions.append(output)
    
    with open('skin\\best_estimators_I\\best_estimator_BKL_VASC_85.pkl', 'rb') as f:
        classifier18 = pickle.load(f)
    output = classifier18.predict(data)
    if output == 0:
        output = 'benign keratosis-like lesion'
    elif output == 1:
        output = 'vascular lesion'
    predictions.append(output)
    
    with open('skin\\best_estimators_I\\best_estimator_DF_MEL_83.pkl', 'rb') as f:
        classifier19 = pickle.load(f)
    output = classifier19.predict(data)
    if output == 0:
        output = 'dermatofibroma'
    elif output == 1:
        output = 'melanoma'
    predictions.append(output)
    
    with open('skin\\best_estimators_I\\best_estimator_DF_NV_86.pkl', 'rb') as f:
        classifier20 = pickle.load(f)
    output = classifier20.predict(data)
    if output == 0:
        output = 'dermatofibroma'
    elif output == 1:
        output = 'nevus'
    predictions.append(output)
    
    with open('skin\\best_estimators_I\\best_estimator_DF_SCC_72.pkl', 'rb') as f:
        classifier21 = pickle.load(f)
    output = classifier21.predict(data)
    if output == 0:
        output = 'dermatofibroma'
    elif output == 1:
        output = 'squamous cell carcinoma'
    predictions.append(output)
    
    with open('skin\\best_estimators_I\\best_estimator_DF_VASC_88.pkl', 'rb') as f:
        classifier22 = pickle.load(f)
    output = classifier22.predict(data)
    if output == 0:
        output = 'dermatofibroma'
    elif output == 1:
        output = 'vascular lesion'
    predictions.append(output)
    
    with open('skin\\best_estimators_I\\best_estimator_MEL_NV_73.pkl', 'rb') as f:
        classifier23 = pickle.load(f)
    output = classifier23.predict(data)
    if output == 0:
        output = 'melanoma'
    elif output == 1:
        output = 'nevus'
    predictions.append(output)
    
    with open('skin\\best_estimators_I\\best_estimator_MEL_SCC_78.pkl', 'rb') as f:
        classifier24 = pickle.load(f)
    output = classifier24.predict(data)
    if output == 0:
        output = 'melanoma'
    elif output == 1:
        output = 'squamous cell carcinoma'
    predictions.append(output)
    
    with open('skin\\best_estimators_I\\best_estimator_MEL_VASC_85.pkl', 'rb') as f:
        classifier25 = pickle.load(f)
    output = classifier25.predict(data)
    if output == 0:
        output = 'melanoma'
    elif output == 1:
        output = 'vascular lesion'
    predictions.append(output)
    
    
    with open('skin\\best_estimators_I\\best_estimator_NV_SCC_87.pkl', 'rb') as f:
        classifier26 = pickle.load(f)
    output = classifier26.predict(data)
    if output == 0:
        output = 'nevus'
    elif output == 1:
        output = 'squamous cell carcinoma'
    predictions.append(output)
    
    with open('skin\\best_estimators_I\\best_estimator_NV_VASC_86.pkl', 'rb') as f:
        classifier27 = pickle.load(f)
    output = classifier27.predict(data)
    if output == 0:
        output = 'nevus'
    elif output == 1:
        output = 'vascular lesion'
    predictions.append(output)
    
    with open('skin\\best_estimators_I\\best_estimator_SCC_VASC_85.pkl', 'rb') as f:
        classifier28 = pickle.load(f)
    output = classifier28.predict(data)
    if output == 0:
        output = 'squamous cell carcinoma'
    elif output == 1:
        output = 'vascular lesion'
    predictions.append(output)

    max = 0
    result = predictions[0]
    for output in predictions:
        frequency = op.countOf(predictions, output)
        if frequency > max:
            max = frequency
            result = output
            
    sorted_predictions = Counter(predictions).most_common()
    print(sorted_predictions)
    print()
    
    if result =="actinic keratosis":
        result = "kératose actinique"
    if result =="basal cell carcinoma":
        result = "carcinome basocellulaire"
    if result =="benign keratosis-like lesion":
        result = "kératose bénigne"
    if result =="dermatofibroma":
        result = "dermatofibrome"
    if result =="melanoma":
        result = "mélanome"
    if result =="nevus":
        result = "nævus mélanocytaire"
    if result =="squamous cell carcinoma":
        result = "carcinome squameux"
    if result =="vascular lesion":
        result = "lésion vasculaire"
    
    
    return f"La lésion sur la photo appartient à la classe {result}."


def SvmPredictor_II(image_path):
    data = []
    image = imread(image_path)
    img = resize(image, (224, 224))
    data.append(img.flatten())
    data = np.asarray(data)

    predictions = []
    # accuracy score moyen de SvmPredictor_II = 79,07 %
    with open('skin\\best_estimators_II\\best_estimator_AK_BCC_63.pkl', 'rb') as f:
        classifier1 = pickle.load(f)
    output = classifier1.predict(data)
    if output == 0:
        output = 'actinic keratosis'
    elif output == 1:
        output = 'basal cell carcinoma'
    predictions.append(output)
    
    with open('skin\\best_estimators_II\\best_estimator_AK_BKL_71.pkl', 'rb') as f:
        classifier2 = pickle.load(f)
    output = classifier2.predict(data)
    if output == 0:
        output = 'actinic keratosis'
    elif output == 1:
        output = 'benign keratosis-like lesion'
    predictions.append(output)
    
    with open('skin\\best_estimators_II\\best_estimator_AK_DF_75.pkl', 'rb') as f:
        classifier3 = pickle.load(f)
    output = classifier3.predict(data)
    if output == 0:
        output = 'actinic keratosis'
    elif output == 1:
        output = 'dermatofibroma'
    predictions.append(output)
    
    with open('skin\\best_estimators_II\\best_estimator_AK_MEL_83.pkl', 'rb') as f:
        classifier4 = pickle.load(f)
    output = classifier4.predict(data)
    if output == 0:
        output = 'actinic keratosis'
    elif output == 1:
        output = 'melanoma'
    predictions.append(output)
    
    with open('skin\\best_estimators_II\\best_estimator_AK_NV_88.pkl', 'rb') as f:
        classifier5 = pickle.load(f)
    output = classifier5.predict(data)
    if output == 0:
        output = 'actinic keratosis'
    elif output == 1:
        output = 'nevus'
    predictions.append(output)
    
    with open('skin\\best_estimators_II\\best_estimator_AK_SCC_67.pkl', 'rb') as f:
        classifier6 = pickle.load(f)
    output = classifier6.predict(data)
    if output == 0:
        output = 'actinic keratosis'
    elif output == 1:
        output = 'squamous cell carcinoma'
    predictions.append(output)
    
    with open('skin\\best_estimators_II\\best_estimator_AK_VASC_87.pkl', 'rb') as f:
        classifier7 = pickle.load(f)
    output = classifier7.predict(data)
    if output == 0:
        output = 'actinic keratosis'
    elif output == 1:
        output = 'vascular lesion'
    predictions.append(output)
    
    with open('skin\\best_estimators_II\\best_estimator_BCC_BKL_72.pkl', 'rb') as f:
        classifier8 = pickle.load(f)
    output = classifier8.predict(data)
    if output == 0:
        output = 'basal cell carcinoma'
    elif output == 1:
        output = 'benign keratosis-like lesion'
    predictions.append(output)
    
    with open('skin\\best_estimators_II\\best_estimator_BCC_DF_77.pkl', 'rb') as f:
        classifier9 = pickle.load(f)
    output = classifier9.predict(data)
    if output == 0:
        output = 'basal cell carcinoma'
    elif output == 1:
        output = 'dermatofibroma'
    predictions.append(output)
    
    with open('skin\\best_estimators_II\\best_estimator_BCC_MEL_80.pkl', 'rb') as f:
        classifier10 = pickle.load(f)
    output = classifier10.predict(data)
    if output == 0:
        output = 'basal cell carcinoma'
    elif output == 1:
        output = 'melanoma'
    predictions.append(output)
    
    with open('skin\\best_estimators_II\\best_estimator_BCC_NV_86.pkl', 'rb') as f:
        classifier11 = pickle.load(f)
    output = classifier11.predict(data)
    if output == 0:
        output = 'basal cell carcinoma'
    elif output == 1:
        output = 'nevus'
    predictions.append(output)
    
    with open('skin\\best_estimators_II\\best_estimator_BCC_SCC_77.pkl', 'rb') as f:
        classifier12 = pickle.load(f)
    output = classifier12.predict(data)
    if output == 0:
        output = 'basal cell carcinoma'
    elif output == 1:
        output = 'squamous cell carcinoma'
    predictions.append(output)
    
    with open('skin\\best_estimators_II\\best_estimator_BCC_VASC_83.pkl', 'rb') as f:
        classifier13 = pickle.load(f)
    output = classifier13.predict(data)
    if output == 0:
        output = 'basal cell carcinoma'
    elif output == 1:
        output = 'vascular lesion'
    predictions.append(output)
    
    with open('skin\\best_estimators_II\\best_estimator_BKL_DF_72.pkl', 'rb') as f:
        classifier14 = pickle.load(f)
    output = classifier14.predict(data)
    if output == 0:
        output = 'benign keratosis-like lesion'
    elif output == 1:
        output = 'dermatofibroma'
    predictions.append(output)
    
    with open('skin\\best_estimators_II\\best_estimator_BKL_MEL_71.pkl', 'rb') as f:
        classifier15 = pickle.load(f)
    output = classifier15.predict(data)
    if output == 0:
        output = 'benign keratosis-like lesion'
    elif output == 1:
        output = 'melanoma'
    predictions.append(output)
    
    with open('skin\\best_estimators_II\\best_estimator_BKL_NV_72.pkl', 'rb') as f:
        classifier16 = pickle.load(f)
    output = classifier16.predict(data)
    if output == 0:
        output = 'benign keratosis-like lesion'
    elif output == 1:
        output = 'nevus'
    predictions.append(output)
    
    with open('skin\\best_estimators_II\\best_estimator_BKL_SCC_78.pkl', 'rb') as f:
        classifier17 = pickle.load(f)
    output = classifier17.predict(data)
    if output == 0:
        output = 'benign keratosis-like lesion'
    elif output == 1:
        output = 'squamous cell carcinoma'
    predictions.append(output)
    
    with open('skin\\best_estimators_II\\best_estimator_BKL_VASC_85.pkl', 'rb') as f:
        classifier18 = pickle.load(f)
    output = classifier18.predict(data)
    if output == 0:
        output = 'benign keratosis-like lesion'
    elif output == 1:
        output = 'vascular lesion'
    predictions.append(output)
    
    with open('skin\\best_estimators_II\\best_estimator_DF_MEL_85.pkl', 'rb') as f:
        classifier19 = pickle.load(f)
    output = classifier19.predict(data)
    if output == 0:
        output = 'dermatofibroma'
    elif output == 1:
        output = 'melanoma'
    predictions.append(output)
    
    with open('skin\\best_estimators_II\\best_estimator_DF_NV_82.pkl', 'rb') as f:
        classifier20 = pickle.load(f)
    output = classifier20.predict(data)
    if output == 0:
        output = 'dermatofibroma'
    elif output == 1:
        output = 'nevus'
    predictions.append(output)
    
    with open('skin\\best_estimators_II\\best_estimator_DF_SCC_76.pkl', 'rb') as f:
        classifier21 = pickle.load(f)
    output = classifier21.predict(data)
    if output == 0:
        output = 'dermatofibroma'
    elif output == 1:
        output = 'squamous cell carcinoma'
    predictions.append(output)
    
    with open('skin\\best_estimators_II\\best_estimator_DF_VASC_86.pkl', 'rb') as f:
        classifier22 = pickle.load(f)
    output = classifier22.predict(data)
    if output == 0:
        output = 'dermatofibroma'
    elif output == 1:
        output = 'vascular lesion'
    predictions.append(output)
    
    with open('skin\\best_estimators_II\\best_estimator_MEL_NV_77.pkl', 'rb') as f:
        classifier23 = pickle.load(f)
    output = classifier23.predict(data)
    if output == 0:
        output = 'melanoma'
    elif output == 1:
        output = 'nevus'
    predictions.append(output)
    
    with open('skin\\best_estimators_II\\best_estimator_MEL_SCC_77.pkl', 'rb') as f:
        classifier24 = pickle.load(f)
    output = classifier24.predict(data)
    if output == 0:
        output = 'melanoma'
    elif output == 1:
        output = 'squamous cell carcinoma'
    predictions.append(output)
    
    with open('skin\\best_estimators_II\\best_estimator_MEL_VASC_86.pkl', 'rb') as f:
        classifier25 = pickle.load(f)
    output = classifier25.predict(data)
    if output == 0:
        output = 'melanoma'
    elif output == 1:
        output = 'vascular lesion'
    predictions.append(output)
    
    
    with open('skin\\best_estimators_II\\best_estimator_NV_SCC_89.pkl', 'rb') as f:
        classifier26 = pickle.load(f)
    output = classifier26.predict(data)
    if output == 0:
        output = 'nevus'
    elif output == 1:
        output = 'squamous cell carcinoma'
    predictions.append(output)
    
    with open('skin\\best_estimators_II\\best_estimator_NV_VASC_84.pkl', 'rb') as f:
        classifier27 = pickle.load(f)
    output = classifier27.predict(data)
    if output == 0:
        output = 'nevus'
    elif output == 1:
        output = 'vascular lesion'
    predictions.append(output)
    
    with open('skin\\best_estimators_II\\best_estimator_SCC_VASC_85.pkl', 'rb') as f:
        classifier28 = pickle.load(f)
    output = classifier28.predict(data)
    if output == 0:
        output = 'squamous cell carcinoma'
    elif output == 1:
        output = 'vascular lesion'
    predictions.append(output)
    

    max = 0
    result = predictions[0]
    for output in predictions:
        frequency = op.countOf(predictions, output)
        if frequency > max:
            max = frequency
            result = output
            
    sorted_predictions = Counter(predictions).most_common()
    print(sorted_predictions)
    print()
        
    if result =="actinic keratosis":
        result = "kératose actinique"
    if result =="basal cell carcinoma":
        result = "carcinome basocellulaire"
    if result =="benign keratosis-like lesion":
        result = "kératose bénigne"
    if result =="dermatofibroma":
        result = "dermatofibrome"
    if result =="melanoma":
        result = "mélanome"
    if result =="nevus":
        result = "nævus mélanocytaire"
    if result =="squamous cell carcinoma":
        result = "carcinome squameux"
    if result =="vascular lesion":
        result = "lésion vasculaire"
    
    return f"La lésion sur la photo appartient à la classe {result}."



def SvmPredictor_III(image_path):
    data = []
    image = imread(image_path)
    img = resize(image, (224, 224))
    data.append(img.flatten())
    data = np.asarray(data)

    predictions = []
    with open('skin\\best_estimators_III\\best_estimator_AK_BCC_68.pkl', 'rb') as f:
        classifier1 = pickle.load(f)
    output = classifier1.predict(data)
    if output == 0:
        output = 'actinic keratosis'
    elif output == 1:
        output = 'basal cell carcinoma'
    predictions.append(output)
    
    with open('skin\\best_estimators_III\\best_estimator_AK_BKL_76.pkl', 'rb') as f:
        classifier2 = pickle.load(f)
    output = classifier2.predict(data)
    if output == 0:
        output = 'actinic keratosis'
    elif output == 1:
        output = 'benign keratosis-like lesion'
    predictions.append(output)
    
    with open('skin\\best_estimators_III\\best_estimator_AK_DF_78.pkl', 'rb') as f:
        classifier3 = pickle.load(f)
    output = classifier3.predict(data)
    if output == 0:
        output = 'actinic keratosis'
    elif output == 1:
        output = 'dermatofibroma'
    predictions.append(output)
    
    with open('skin\\best_estimators_III\\best_estimator_AK_MEL_85.pkl', 'rb') as f:
        classifier4 = pickle.load(f)
    output = classifier4.predict(data)
    if output == 0:
        output = 'actinic keratosis'
    elif output == 1:
        output = 'melanoma'
    predictions.append(output)
    
    with open('skin\\best_estimators_III\\best_estimator_AK_NV_93.pkl', 'rb') as f:
        classifier5 = pickle.load(f)
    output = classifier5.predict(data)
    if output == 0:
        output = 'actinic keratosis'
    elif output == 1:
        output = 'nevus'
    predictions.append(output)
    
    with open('skin\\best_estimators_III\\best_estimator_AK_SCC_73.pkl', 'rb') as f:
        classifier6 = pickle.load(f)
    output = classifier6.predict(data)
    if output == 0:
        output = 'actinic keratosis'
    elif output == 1:
        output = 'squamous cell carcinoma'
    predictions.append(output)
    
    with open('skin\\best_estimators_III\\best_estimator_AK_VASC_88.pkl', 'rb') as f:
        classifier7 = pickle.load(f)
    output = classifier7.predict(data)
    if output == 0:
        output = 'actinic keratosis'
    elif output == 1:
        output = 'vascular lesion'
    predictions.append(output)
    
    with open('skin\\best_estimators_III\\best_estimator_BCC_BKL_72.pkl', 'rb') as f:
        classifier8 = pickle.load(f)
    output = classifier8.predict(data)
    if output == 0:
        output = 'basal cell carcinoma'
    elif output == 1:
        output = 'benign keratosis-like lesion'
    predictions.append(output)
    
    with open('skin\\best_estimators_III\\best_estimator_BCC_DF_76.pkl', 'rb') as f:
        classifier9 = pickle.load(f)
    output = classifier9.predict(data)
    if output == 0:
        output = 'basal cell carcinoma'
    elif output == 1:
        output = 'dermatofibroma'
    predictions.append(output)
    
    with open('skin\\best_estimators_III\\best_estimator_BCC_MEL_81.pkl', 'rb') as f:
        classifier10 = pickle.load(f)
    output = classifier10.predict(data)
    if output == 0:
        output = 'basal cell carcinoma'
    elif output == 1:
        output = 'melanoma'
    predictions.append(output)
    
    with open('skin\\best_estimators_III\\best_estimator_BCC_NV_85.pkl', 'rb') as f:
        classifier11 = pickle.load(f)
    output = classifier11.predict(data)
    if output == 0:
        output = 'basal cell carcinoma'
    elif output == 1:
        output = 'nevus'
    predictions.append(output)
    
    with open('skin\\best_estimators_III\\best_estimator_BCC_SCC_71.pkl', 'rb') as f:
        classifier12 = pickle.load(f)
    output = classifier12.predict(data)
    if output == 0:
        output = 'basal cell carcinoma'
    elif output == 1:
        output = 'squamous cell carcinoma'
    predictions.append(output)
    
    with open('skin\\best_estimators_III\\best_estimator_BCC_VASC_82.pkl', 'rb') as f:
        classifier13 = pickle.load(f)
    output = classifier13.predict(data)
    if output == 0:
        output = 'basal cell carcinoma'
    elif output == 1:
        output = 'vascular lesion'
    predictions.append(output)
    
    with open('skin\\best_estimators_III\\best_estimator_BKL_DF_75.pkl', 'rb') as f:
        classifier14 = pickle.load(f)
    output = classifier14.predict(data)
    if output == 0:
        output = 'benign keratosis-like lesion'
    elif output == 1:
        output = 'dermatofibroma'
    predictions.append(output)
    
    with open('skin\\best_estimators_III\\best_estimator_BKL_MEL_72.pkl', 'rb') as f:
        classifier15 = pickle.load(f)
    output = classifier15.predict(data)
    if output == 0:
        output = 'benign keratosis-like lesion'
    elif output == 1:
        output = 'melanoma'
    predictions.append(output)
    
    with open('skin\\best_estimators_III\\best_estimator_BKL_NV_77.pkl', 'rb') as f:
        classifier16 = pickle.load(f)
    output = classifier16.predict(data)
    if output == 0:
        output = 'benign keratosis-like lesion'
    elif output == 1:
        output = 'nevus'
    predictions.append(output)
    
    with open('skin\\best_estimators_III\\best_estimator_BKL_SCC_80.pkl', 'rb') as f:
        classifier17 = pickle.load(f)
    output = classifier17.predict(data)
    if output == 0:
        output = 'benign keratosis-like lesion'
    elif output == 1:
        output = 'squamous cell carcinoma'
    predictions.append(output)
    
    with open('skin\\best_estimators_III\\best_estimator_BKL_VASC_82.pkl', 'rb') as f:
        classifier18 = pickle.load(f)
    output = classifier18.predict(data)
    if output == 0:
        output = 'benign keratosis-like lesion'
    elif output == 1:
        output = 'vascular lesion'
    predictions.append(output)
    
    with open('skin\\best_estimators_III\\best_estimator_DF_MEL_82.pkl', 'rb') as f:
        classifier19 = pickle.load(f)
    output = classifier19.predict(data)
    if output == 0:
        output = 'dermatofibroma'
    elif output == 1:
        output = 'melanoma'
    predictions.append(output)
    
    with open('skin\\best_estimators_III\\best_estimator_DF_NV_89.pkl', 'rb') as f:
        classifier20 = pickle.load(f)
    output = classifier20.predict(data)
    if output == 0:
        output = 'dermatofibroma'
    elif output == 1:
        output = 'nevus'
    predictions.append(output)
    
    with open('skin\\best_estimators_III\\best_estimator_DF_SCC_84.pkl', 'rb') as f:
        classifier21 = pickle.load(f)
    output = classifier21.predict(data)
    if output == 0:
        output = 'dermatofibroma'
    elif output == 1:
        output = 'squamous cell carcinoma'
    predictions.append(output)
    
    with open('skin\\best_estimators_III\\best_estimator_DF_VASC_87.pkl', 'rb') as f:
        classifier22 = pickle.load(f)
    output = classifier22.predict(data)
    if output == 0:
        output = 'dermatofibroma'
    elif output == 1:
        output = 'vascular lesion'
    predictions.append(output)
    
    with open('skin\\best_estimators_III\\best_estimator_MEL_NV_81.pkl', 'rb') as f:
        classifier23 = pickle.load(f)
    output = classifier23.predict(data)
    if output == 0:
        output = 'melanoma'
    elif output == 1:
        output = 'nevus'
    predictions.append(output)
    
    with open('skin\\best_estimators_III\\best_estimator_MEL_SCC_86.pkl', 'rb') as f:
        classifier24 = pickle.load(f)
    output = classifier24.predict(data)
    if output == 0:
        output = 'melanoma'
    elif output == 1:
        output = 'squamous cell carcinoma'
    predictions.append(output)
    
    with open('skin\\best_estimators_III\\best_estimator_MEL_VASC_83.pkl', 'rb') as f:
        classifier25 = pickle.load(f)
    output = classifier25.predict(data)
    if output == 0:
        output = 'melanoma'
    elif output == 1:
        output = 'vascular lesion'
    predictions.append(output)
    
    
    with open('skin\\best_estimators_III\\best_estimator_NV_SCC_92.pkl', 'rb') as f:
        classifier26 = pickle.load(f)
    output = classifier26.predict(data)
    if output == 0:
        output = 'nevus'
    elif output == 1:
        output = 'squamous cell carcinoma'
    predictions.append(output)
    
    with open('skin\\best_estimators_III\\best_estimator_NV_VASC_86.pkl', 'rb') as f:
        classifier27 = pickle.load(f)
    output = classifier27.predict(data)
    if output == 0:
        output = 'nevus'
    elif output == 1:
        output = 'vascular lesion'
    predictions.append(output)
    
    with open('skin\\best_estimators_III\\best_estimator_SCC_VASC_82.pkl', 'rb') as f:
        classifier28 = pickle.load(f)
    output = classifier28.predict(data)
    if output == 0:
        output = 'squamous cell carcinoma'
    elif output == 1:
        output = 'vascular lesion'
    predictions.append(output)
    

    max = 0
    result = predictions[0]
    for output in predictions:
        frequency = op.countOf(predictions, output)
        if frequency > max:
            max = frequency
            result = output
            
    sorted_predictions = Counter(predictions).most_common()
    print(sorted_predictions)
    
    if result =="actinic keratosis":
        result = "kératose actinique"
    if result =="basal cell carcinoma":
        result = "carcinome basocellulaire"
    if result =="benign keratosis-like lesion":
        result = "kératose bénigne"
    if result =="dermatofibroma":
        result = "dermatofibrome"
    if result =="melanoma":
        result = "mélanome"
    if result =="nevus":
        result = "nævus mélanocytaire"
    if result =="squamous cell carcinoma":
        result = "carcinome squameux"
    if result =="vascular lesion":
        result = "lésion vasculaire"
    
    return f"La lésion sur la photo appartient à la classe {result}."


def wisdom_of_the_crowd(image_path):
    occurrences = []
    result_1 = SvmPredictor_I(image_path)
    occurrences.append(result_1)
    result_2 = SvmPredictor_II(image_path)
    occurrences.append(result_2)
    result_3 = SvmPredictor_III(image_path)
    occurrences.append(result_3)
   
    counter = Counter(occurrences)
    most_frequent_output = counter.most_common(1)[0][0]
    return most_frequent_output


# print(majority_wins('C:\\Users\\frede\\OneDrive\\Bureau\\Isic_skin_cancer_2019\\VASC\\ISIC_0024706.jpg'))
  










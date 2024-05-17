import cv2

# Model görüntüsünü renkli olarak yükle
model_image = cv2.imread('2.jpg', cv2.IMREAD_COLOR)

# Model görüntüsü yoksa hata mesajı ver ve çık
if model_image is None:
    print("Model görüntüsü yüklenemedi!")
    exit()

# Kamera yakalamasını başlat
cap = cv2.VideoCapture(0)  # 0: ilk kamera

# ORB (Oriented FAST and Rotated BRIEF) özellik dedektörü ve eşleştirici oluştur
orb = cv2.ORB_create()
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Model görüntüsündeki ORB özelliklerini bul
model_keypoints, model_descriptors = orb.detectAndCompute(model_image, None)

while True:
    # Görüntü yakalaması yap
    ret, frame = cap.read()

    # Görüntüdeki ORB özelliklerini bul
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_keypoints, frame_descriptors = orb.detectAndCompute(gray_frame, None)

    # Eşleşen özellikleri bul
    matches = matcher.match(model_descriptors, frame_descriptors)

    # Listeye dönüştür
    matches = list(matches)

    # En iyi eşleşmeleri sırala
    matches.sort(key=lambda x: x.distance)

    # İlk N eşleşmeyi al (örneğin, en iyi 10 eşleşme)
    top_matches = matches[:10]

    # Eşleşme çizgilerini çiz ve ekrana yazdır
    matched_image = cv2.drawMatches(model_image, model_keypoints, frame, frame_keypoints, top_matches, None)
    cv2.imshow('Matched Image', matched_image)

    # Çıkış için 'q' tuşuna basılmasını bekle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamerayı serbest bırak ve pencereleri kapat
cap.release()
cv2.destroyAllWindows()



import coremltools as ct

# 모델 로드
model = ct.models.MLModel('sauron_newall.mlmodel')

# 모델 입력 사양 출력
print(model.get_spec().description.input[0])

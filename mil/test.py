import slideflow as sf
from slideflow.model import build_feature_extractor


main_path = '/nfs/home/users/fshahi/Projects/Histomorphological-Phenotype-Learning'
project_path = '{}/{}'.format(main_path, 'mil')
# P = sf.create_project(
#     root= project_path,
#     name='mil',
# )
P = sf.load_project(project_path)
print(P)

ctranspath = build_feature_extractor('retccl', tile_px=224)
print(ctranspath)
print('models:', sf.model.list_extractors())
for model in sf.model.list_extractors():
    print(model, sf.model.is_extractor(model))


resnet50 = build_feature_extractor(
    'resnet50_imagenet',
    tile_px=299
)
print(resnet50)

class BaseConfig:
    name = None
    text = None
    character_images = []
    character_names = []
    cache_context = None
    cache_characters = []
    seed = 100


class CatConfig1(BaseConfig):
    name = "Cat Hot Dog (basic)"
    text = "cat_hot_dog.jpg"


class CatConfig2(BaseConfig):
    name = "Cat Hot Dog (Bird only)"
    text = "cat_hot_dog.jpg"
    character_images = ["blue_bird.jpg"]


class CatConfig3(BaseConfig):
    name = "Cat Hot Dog (both characters)"
    text = "cat_hot_dog.jpg"
    character_images = ["orange_cat.jpg", "blue_bird.jpg"]


class CatConfig4(BaseConfig):
    name = "Cat Hot Dog Ending (using cache)"
    text = "cat_hot_dog_ending.jpg"
    cache_context = "Tony the Cat Hot Dog Story"
    cache_characters = ["Tony", "Bird"]


class CatConfig5(BaseConfig):
    name = "Cat Hot Dog Ending (no cache)"
    text = "cat_hot_dog_ending.jpg"


class JianboConfig(BaseConfig):
    name = "Jianbo Hot Dog (named character)"
    text = "jianbo_hot_dog.jpg"
    character_images = ["penguin.jpg"]
    character_names = ["Jianbo"]


class DogConfig1(BaseConfig):
    name = "Dog Armadillo (basic)"
    text = "dog_armadillo.jpg"


class DogConfig2(BaseConfig):
    name = "Dog Armadillo (Armadillo only)"
    text = "dog_armadillo.jpg"
    character_images = ["roadkill.jpg"]
    character_names = ["Amelia"]


class DogConfig3(BaseConfig):
    name = "Dog Armadillo (both characters)"
    text = "dog_armadillo.jpg"
    character_images = ["roadkill.jpg", "dalmation.png"]


class HorseConfig1(BaseConfig):
    name = "Horse Giraffe (basic)"
    text = "horse_giraffe.jpg"


class HorseConfig2(BaseConfig):
    name = "Horse Giraffe (horse only)"
    text = "horse_giraffe.jpg"
    character_images = ["horse.jpg"]
    character_nmaes = ["Lucy"]
    seed = 200


class HorseConfig3(BaseConfig):
    name = "Horse Giraffe (both characters)"
    text = "horse_giraffe.jpg"
    character_images = ["horse.jpg", "giraffe.jpg"]
    seed = 200


class TrainConfig1(BaseConfig):
    name = "Train Part 1"
    text = "train_part_1.jpg"
    seed = 300


class TrainConfig2(BaseConfig):
    name = "Train Part 2"
    text = "train_part_2.jpg"
    seed = 300


class TrainConfig3(BaseConfig):
    name = "Train Part 2 (with context)"
    text = "train_part_2.jpg"
    cache_context = "Train Story Part 1"
    seed = 300


class TrainConfig4(BaseConfig):
    name = "Train Part 3"
    text = "train_part_3.jpg"
    seed = 200


class TrainConfig5(BaseConfig):
    name = "Train Part 3 (with context)"
    text = "train_part_3.jpg"
    cache_context = "Train Story Part 2"
    seed = 200


class AnimalsConfig1(BaseConfig):
    name = "Animals Playing Catch (basic)"
    text = "animals_catch.jpg"
    seed = 400


class AnimalsConfig2(BaseConfig):
    name = "Animals Playing Catch (all characters)"
    text = "animals_catch.jpg"
    character_images = ["orange_cat.jpg", "dalmation.png", "blue_bird.jpg", "horse.jpg"]
    seed = 400

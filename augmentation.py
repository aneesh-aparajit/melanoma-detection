import Augmentor as ag
p = ag.Pipeline("../data/train/")
p.rotate90(probability=0.5)
p.zoom(probability=0.3, min_factor=0.8, max_factor=1.5)
p.flip_left_right(probability=0.8)
p.flip_top_bottom(probability=0.3)
p.random_brightness(probability=0.3, min_factor=0.3, max_factor=1.2)
p.sample(10000)
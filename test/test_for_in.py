boxes = [[0.5,0.45,1.2,4],
        [0.6,0.35,1.4,3],
        [0.2,0.46,1.5,4],
        [0.4,0.45,1.3,5]]

for box in boxes:
    box = box[2:]
    assert len(box) == 4
    print(box)

print(box)
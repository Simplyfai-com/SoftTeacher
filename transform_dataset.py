import os
import json

ORIGIN_PATH = "./full_treadmill_pytorch_dataset"
DESTINATION_PATH = "./data/coco/annotations"

LICENCES = [
    {
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
    },
    {
        "url": "http://creativecommons.org/licenses/by-nc/2.0/",
        "id": 2,
        "name": "Attribution-NonCommercial License",
    },
    {
        "url": "http://creativecommons.org/licenses/by-nc-nd/2.0/",
        "id": 3,
        "name": "Attribution-NonCommercial-NoDerivs License",
    },
    {
        "url": "http://creativecommons.org/licenses/by/2.0/",
        "id": 4,
        "name": "Attribution License",
    },
    {
        "url": "http://creativecommons.org/licenses/by-sa/2.0/",
        "id": 5,
        "name": "Attribution-ShareAlike License",
    },
    {
        "url": "http://creativecommons.org/licenses/by-nd/2.0/",
        "id": 6,
        "name": "Attribution-NoDerivs License",
    },
    {
        "url": "http://flickr.com/commons/usage/",
        "id": 7,
        "name": "No known copyright restrictions",
    },
    {
        "url": "http://www.usa.gov/copyright.shtml",
        "id": 8,
        "name": "United States Government Work",
    },
]

CATEGORIES = [{"supercategory": "box", "id": 1, "name": "box"}]


def main():
    for folder in os.listdir(ORIGIN_PATH):
        print(f"Getting into {folder}")
        continue
        path = f"{ORIGIN_PATH}/{folder}/images"
        images = [
            {"id": image.replace(".jpg", ""), "file_name": image}
            for image in os.listdir(path)
        ]
        annotations = []
        images_info = []
        for image in images:
            image_id = image.get("id")
            with open(
                f"{ORIGIN_PATH}/{folder}/annotations/{image_id}.json"
            ) as annotations_file:
                image_annotations = json.load(annotations_file)
                bboxes = image_annotations.get("bboxes", [])[0]
                _, _, bbox_width, bbox_height = bboxes
                # bbox_width = xmax - xmin + 1
                # bbox_height = ymax - ymin + 1
                if folder == "test":
                    image_id = int.from_bytes(image_id.encode(), "big")
                image_annotations.update(
                    {
                        "image_id": image_id,
                        "id": image_id,
                        "category_id": 1,
                        "width": bbox_width,
                        "height": bbox_height,
                        "area": bbox_width * bbox_height,
                        "iscrowd": 0,
                        "bbox": bboxes,
                    }
                )
                annotations.append(image_annotations)
                images_info.append(
                    {
                        "id": image_id,
                        "licence": 1,
                        "file_name": image.get("file_name"),
                        "width": bbox_width,
                        "height": bbox_height,
                    }
                )

        annotations = {
            "licenses": LICENCES,
            "categories": CATEGORIES,
            "images": images_info,
            "annotations": annotations,
        }
        out_path = f"{DESTINATION_PATH}/instances_{folder}2017.json"
        with open(out_path, "w") as save_file:
            json.dump(annotations, save_file)
        if folder == "test":
            out_path = f"{DESTINATION_PATH}/instances_val2017.json"
            with open(out_path, "w") as save_file:
                json.dump(annotations, save_file)


if __name__ == "__main__":
    main()

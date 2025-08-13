PROJECT_NAME := aiexpert
IMAGE_NAME := seunguk/${PROJECT_NAME}
SHM_SIZE := 64gb
DIR ?=./user0
GPU_ID ?= 0
PORT ?= 9000
USER_ID ?= 0

all: build run-user0 run-user1 run-user2 run-user3 run-user4 run-user5 run-user6 run-user7

build:
	docker build \
		--tag ${IMAGE_NAME}:latest \
		-f Dockerfile .

run:
	@if [ ! -d ${DIR} ]; then \
		mkdir ${DIR}; \
	fi
	docker run \
		-it \
		--rm \
		--shm-size 64gb \
		--workdir="/app" \
		--gpus "device=${GPU_ID}" \
		--volume="${DIR}:/app" \
		--volume="./data:/data" \
		-p ${PORT}:8888 \
		${IMAGE_NAME}:latest

download-data:
	wget "https://cvg-data.inf.ethz.ch/openscene/data/scannet_processed/scannet_3d.zip" -O ./data/scannet_3d.zip
	unzip ./data/scannet_3d.zip -d ./data

copy_materials_single:
	mkdir ./user${USER_ID}
	cp materials/3DPerception.ipynb ./user${USER_ID}
	cp materials/*.py ./user${USER_ID}

copy-materials:
	$(MAKE) copy_materials_single USER_ID=0
	$(MAKE) copy_materials_single USER_ID=1
	$(MAKE) copy_materials_single USER_ID=2
	$(MAKE) copy_materials_single USER_ID=3
	$(MAKE) copy_materials_single USER_ID=4
	$(MAKE) copy_materials_single USER_ID=5
	$(MAKE) copy_materials_single USER_ID=6
	$(MAKE) copy_materials_single USER_ID=7

run-user0:
	$(MAKE) run DIR=user0 GPU_ID=0 PORT=9000

run-user1:
	$(MAKE) run DIR=user1 GPU_ID=1 PORT=9001

run-user2:
	$(MAKE) run DIR=user2 GPU_ID=2 PORT=9002

run-user3:
	$(MAKE) run DIR=user3 GPU_ID=3 PORT=9003

run-user4:
	$(MAKE) run DIR=user4 GPU_ID=4 PORT=9004

run-user5:
	$(MAKE) run DIR=user5 GPU_ID=5 PORT=9005

run-user6:
	$(MAKE) run DIR=user6 GPU_ID=6 PORT=9006

run-user7:
	$(MAKE) run DIR=user7 GPU_ID=7 PORT=9007


.PHONY: run build

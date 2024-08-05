all: build run
build:
	docker build -t coreml-to-pt .
run:
	docker run -it --rm coreml-to-pt
img:
	docker images -a
ps:
	docker ps -a
rm:
	docker rm -f $$(docker ps -aq)
rmi:
	docker rmi -f $$(docker images -a)


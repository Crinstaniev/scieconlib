docker build -t crinstaniev/scieconlib-doc:latest .
docker push crinstaniev/scieconlib-doc:latest

docker container list
docker pull crinstaniev/scieconlib-doc:latest
docker run -e VIRTUAL_HOST=scieconlib.ppsh.su -t -d -p 8000:80 crinstaniev/scieconlib-doc:latest
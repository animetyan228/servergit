FROM node:20

WORKDIR /app

COPY package*.json ./

RUN npm install react-bootstrap bootstrap

COPY . .

RUN npm run build

FROM nginx:alpine

COPY --from=0 /app/build /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]

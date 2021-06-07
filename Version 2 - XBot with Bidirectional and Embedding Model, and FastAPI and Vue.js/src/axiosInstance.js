import axios from "axios";

const instance = axios.create({
  baseURL: "http://localhost:8080",
  withCredentials: true,
  progress: true,
});

function getSingleEndpoint(parameters, endpoint) {
  const options = {
    ...parameters,
  };

  const args = Object.entries(options)
    .map((d) => `${d[0]}=${d[1]}`)
    .join("&");

  return instance.get(`/${endpoint}?${args}`);
}

export { instance, getSingleEndpoint };

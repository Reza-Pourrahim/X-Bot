<template>
  <b-container>
    <div>
      <b-row>
        <b-col>
          <b-jumbotron>
            <template #header>X-Bot</template>
            <template #lead>
              Development of a Model and Data Agnostic Chat Bot for Explaining
              the Decisions of Black Box Classifiers
              <hr />
              <b>
                Prof. Fosca Giannotti<br />
                Prof. Riccardo Guidotti<br />
                Prof. Simone Scardapane<br />
                <br />
                Reza Pourrahim<br />
              </b>
              <br /><br />
              <small>June 2021</small>
            </template>
          </b-jumbotron>
        </b-col>
      </b-row>

      <b-row>
        <b-col md="4">
          <b-form @submit="onSubmit">
            <b-form-group
              id="ig-sepal_length"
              label="Sepal Length:"
              label-for="i-sepal_length"
            >
              <b-form-input
                id="i-sepal_length"
                size="sm"
                v-model="form.sepal_length"
                type="number"
                min="0"
                step="0.1"
                required
              ></b-form-input>
            </b-form-group>
            <b-form-group
              id="ig-sepal_width"
              label="Sepal Width:"
              label-for="i-sepal_width"
            >
              <b-form-input
                id="i-sepal_width"
                size="sm"
                v-model="form.sepal_width"
                type="number"
                min="0"
                step="0.1"
                required
              ></b-form-input>
            </b-form-group>
            <b-form-group
              id="ig-petal_length"
              label="Petal Length:"
              label-for="i-petal_length"
            >
              <b-form-input
                id="i-petal_length"
                size="sm"
                v-model="form.petal_length"
                type="number"
                min="0"
                step="0.1"
                required
              ></b-form-input>
            </b-form-group>
            <b-form-group
              id="ig-petal_width"
              label="Petal Width:"
              label-for="i-petal_width"
            >
              <b-form-input
                id="i-petal_width"
                size="sm"
                v-model="form.petal_width"
                type="number"
                min="0"
                step="0.1"
                required
              ></b-form-input>
            </b-form-group>
            <b-button type="submit" variant="primary" :disabled="invalid"
              >Predict</b-button
            >
            <b-overlay no-wrap :show="invalid"></b-overlay>
          </b-form>
        </b-col>
        <hr />
        <hr />
        <b-col>
          <b-form @submit="onSubmit_chat">
            <b-form-group
              id="ig-user_input"
              label="X-Bot:"
              label-for="i-user_input"
            >
              <b-form-input
                id="i-user_input"
                size="lg"
                placeholder="Ask me your question"
                v-model="form_chat.user_input"
                type="text"
                required
              ></b-form-input>
            </b-form-group>
            <b-button type="submit" variant="primary" :disabled="invalid_chat"
              >Send</b-button
            >
            <b-overlay no-wrap :show="invalid_chat"></b-overlay>
          </b-form>
        </b-col>
        <b-col v-if="explanation" class="position-relative">
          <b-card no-body>
            <b-card-body>
              <h3>
                Class:
                <b-badge> {{ explanation }}</b-badge>
              </h3>
            </b-card-body>
          </b-card>
        </b-col>
      </b-row>
    </div>
  </b-container>
</template>

<script>
// import { getSingleEndpoint } from "@/axiosInstance";
import axios from "axios";

export default {
  name: "Home",
  data() {
    return {
      form: {
        sepal_length: 0,
        sepal_width: 0,
        petal_length: 0,
        petal_width: 0,
      },
      form_chat: {
        user_input: "",
      },
      explanation: null,
      invalid: false,
      invalid_chat: false,
    };
  },
  methods: {
    onSubmit(event) {
      event.preventDefault();
      this.invalid = true;
      // alert(JSON.stringify(this.form));
      axios
        .post("/explain_iris", this.form)
        .then((res) => {
          this.explanation = res.data;
          this.invalid = false;
        })
        .catch((e) => console.log(e));
    },
    onSubmit_chat(event) {
      event.preventDefault();
      this.invalid_chat = true;

      axios
        .get("/chat_bot", this.form_chat)
        .then((res) => {
          this.explanation = res.data;
          this.invalid_chat = false;
        })
        .catch((e) => console.log(e));
    },
  },
};
</script>

<!--npm run lint -- --fix -->

<template>
  <b-container>
    <div id="compas">
      <b-jumbotron
        bg-variant="info"
        text-variant="Secondary"
        border-variant="dark"
      >
        <template #header>Iris</template>
        <template #lead> The Iris Flower dataset </template>
        <hr />
        <br />
        <b-button v-b-toggle.sidebar-1>Input Values</b-button>
        <b-sidebar id="sidebar-1" title="Iris" shadow>
          <div class="px-3 py-2">
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
          </div>
        </b-sidebar>
      </b-jumbotron>
      <b-row cols="2">
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
            <b-button type="submit" variant="primary">Send</b-button>
          </b-form>
        </b-col>
      </b-row>
    </div>
  </b-container>
</template>

<script>
import axios from "axios";

export default {
  name: "Iris",
  data() {
    return {
      form_chat: {
        user_input: "",
      },
      form: {
        sepal_length: 0,
        sepal_width: 0,
        petal_length: 0,
        petal_width: 0,
      },
      invalid: false,
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
  },
};
</script>

<style scoped></style>

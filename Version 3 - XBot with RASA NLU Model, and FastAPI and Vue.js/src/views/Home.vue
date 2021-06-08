<!--npm run lint -- --fix -->

<template>
  <b-container>
    <div>
      <b-row>
        <b-col>
          <b-jumbotron
            bg-variant="info"
            text-variant="Secondary"
            border-variant="dark"
          >
            <template #header>X-Bot</template>
            <template #lead>
              Development of a Model and Data Agnostic Chat Bot for Explaining
              the Decisions of Black Box Classifiers
              <hr />
              Prof. Fosca Giannotti<br />
              Prof. Riccardo Guidotti<br />
              Prof. Simone Scardapane<br />
              <br />
              Reza Pourrahim<br />
              <br /><br />
              <small>June 2021</small>
            </template>
          </b-jumbotron>
        </b-col>
      </b-row>

      <b-row cols="2">
        <b-col>
          <label for="ig-explainer_dataset">Dataset:</label>
          <b-form-select
            id="ig-explainer_dataset"
            v-model="explainer_dataset"
            :options="explainer_dataset_options"
          ></b-form-select>
        </b-col>
        <b-col>
          <label for="ig-explainer_model">Explainer:</label>
          <b-form-select
            id="ig-explainer_model"
            v-model="explainer_model"
            :options="explainer_model_options"
          ></b-form-select>
        </b-col>
      </b-row>
      <br />
      <b-button type="submit" variant="primary" :disabled="invalid"
        >Go</b-button
      >
      <br />
    </div>
  </b-container>
</template>

<script>
// import { getSingleEndpoint } from "@/axiosInstance";
// import axios from "axios";

export default {
  name: "Home",
  data() {
    return {
      explainer_dataset: null,
      explainer_model: null,
      explainer_dataset_options: [
        { value: null, text: "Please select an option" },
        { value: "compas", text: "COMPAS" },
        { value: "iris", text: "Iris" },
      ],
      explainer_model_options: [
        { value: null, text: "Please select an option" },
        { value: "lore", text: "LORE" },
      ],
    };
  },
  methods: {},
};
</script>

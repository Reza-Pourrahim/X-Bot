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
            <template #header>
              <div>
                <b-img
                  fluid
                  left
                  src="https://www.uniroma1.it/sites/default/files/images/logo/sapienza-big.png"
                  alt="Fluid image"
                ></b-img>
                <b-img
                  fluid
                  right
                  src="https://kdd.isti.cnr.it/sites/kdd.isti.cnr.it/files/LogoH100.png"
                  alt="Fluid image"
                ></b-img>
              </div>
              <br />
              X-Bot</template
            >
            <template #lead>
              Development of a Model and Data Agnostic Chat Bot for Explaining
              the Decisions of Black Box Classifiers
              <hr />
              <h5>
                Faculty of Information Engineering, Computer Science and
                Statistics
              </h5>
              <br />
              <h5>Candidate:</h5>
              <h6>Reza Pourrahim</h6>
              <h6>1859334</h6>

              <b-row cols-md="2">
                <b-col>
                  <h5>KDD Lab Advisors:</h5>
                  <h6>Prof. Fosca Giannotti</h6>
                  <h6>Prof. Riccardo Guidotti</h6>
                </b-col>
                <b-col>
                  <h5>Sapienza Universty Advisor:</h5>
                  <h6>Prof. Simone Scardapane</h6>
                </b-col>
              </b-row>
              <br /><br />
              <h6>2020/2021</h6>
            </template>
          </b-jumbotron>
        </b-col>
      </b-row>

      <b-form @submit="onSubmit">
        <b-row cols="3">
          <b-col>
            <b-form-group
              id="ig-explainer_dataset"
              label="Dataset:"
              label-for="ig-explainer_dataset"
            >
              <b-form-select
                id="ig-explainer_dataset"
                v-model="form.explainer_dataset"
                :options="explainer_dataset_options"
                required
              ></b-form-select>
            </b-form-group>
          </b-col>
          <b-col>
            <b-form-group
              id="ig-explainer_model"
              label="Explainer:"
              label-for="ig-explainer_model"
            >
              <b-form-select
                id="ig-explainer_model"
                v-model="form.explainer_model"
                :options="explainer_model_options"
                required
              ></b-form-select>
            </b-form-group>
          </b-col>
          <b-col>
            <br />
            <b-button
              type="submit"
              variant="primary"
              size="lg"
              :disabled="invalid"
              >Go</b-button
            >
          </b-col>
        </b-row>
        <b-overlay no-wrap :show="invalid"></b-overlay>
      </b-form>
      <br /><br /><br />
    </div>
  </b-container>
</template>

<script>
export default {
  name: "Home",
  data() {
    return {
      invalid: false,
      form: {
        explainer_dataset: null,
        explainer_model: null,
      },
      explainer_dataset_options: [
        { value: null, text: "Please select an option" },
        { value: "adult", text: "Adult Income" },
        { value: "compas", text: "COMPAS" },
        { value: "german_credit", text: "German Credit" },
        { value: "iris", text: "Iris" },
        { value: "wine", text: "Wine" },
      ],
      explainer_model_options: [
        { value: null, text: "Please select an option" },
        { value: "lore", text: "LORE - Local Rule-Based Explanations" },
        // {
        //   value: "lime",
        //   text: "LIME - Local Interpretable Model-Agnostic Explanations (not implemented yet)",
        // },
      ],
    };
  },
  methods: {
    onSubmit(event) {
      event.preventDefault();
      this.invalid = true;
      this.$router.push(
        "/" + this.form.explainer_dataset + "_" + this.form.explainer_model
      );
    },
  },
};
</script>

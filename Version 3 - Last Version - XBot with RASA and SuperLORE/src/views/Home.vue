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
        <b-row cols="2">
          <b-col>
            <b-form-group
              id="ig-explainer_dataset"
              label="Data:"
              label-for="ig-explainer_dataset"
            >
              <b-form-select
                id="ig-explainer_dataset"
                v-model="form.data_to_explain"
                :options="data_to_explain_options"
                required
              ></b-form-select>
            </b-form-group>
          </b-col>
          <!--          <b-col>-->
          <!--            <b-form-group-->
          <!--              id="ig-explainer_model"-->
          <!--              label="Model:"-->
          <!--              label-for="ig-explainer_model"-->
          <!--            >-->
          <!--              <b-form-select-->
          <!--                id="ig-explainer_model"-->
          <!--                v-model="form.model_to_explain"-->
          <!--                :options="model_to_explain_options"-->
          <!--                required-->
          <!--              ></b-form-select>-->
          <!--            </b-form-group>-->
          <!--          </b-col>-->
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
        data_to_explain: null,
        // model_to_explain: null,
      },
      data_to_explain_options: [
        { value: null, text: "Please select an option" },
        { value: "adult", text: "Adult Income" },
        { value: "compas", text: "COMPAS" },
        { value: "german_credit", text: "German Credit" },
        { value: "iris", text: "Iris" },
        { value: "wine", text: "Wine" },
      ],
      // model_to_explain_options: [
      //   { value: null, text: "Please select an option" },
      //   { value: "RandomForestClassifier", text: "Random Forest Classifier" },
      //   { value: "SGDClassifier", text: "Stochastic Gradient Descent Classifier" },
      //   { value: "SVC", text: "Support Vector Machine Classifier" },
      // ],
    };
  },
  methods: {
    onSubmit(event) {
      event.preventDefault();
      this.invalid = true;
      this.$router.push("/" + this.form.data_to_explain);
    },
  },
};
</script>

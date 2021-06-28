import Vue from "vue";
import VueRouter from "vue-router";
import Home from "../views/Home.vue";

Vue.use(VueRouter);

const routes = [
  {
    path: "/",
    name: "Home",
    component: Home,
  },
  {
    path: "/compas_lore",
    name: "The COMPAS (Correctional Offender Management Profiling for Alternative Sanctions Dataset",
    // route level code-splitting
    // this generates a separate chunk (about.[hash].js) for this route
    // which is lazy-loaded when the route is visited.
    component: () =>
      import(/* webpackChunkName: "compas_lore" */ "../views/Compas.vue"),
  },
  {
    path: "/iris_lore",
    name: "Iris Flower Dataset",
    // route level code-splitting
    // this generates a separate chunk (about.[hash].js) for this route
    // which is lazy-loaded when the route is visited.
    component: () =>
      import(/* webpackChunkName: "iris_lore" */ "../views/Iris.vue"),
  },
  {
    path: "/adult_lore",
    name: "The Adult Income Dataset",
    // route level code-splitting
    // this generates a separate chunk (about.[hash].js) for this route
    // which is lazy-loaded when the route is visited.
    component: () =>
      import(/* webpackChunkName: "adult_lore" */ "../views/Adult.vue"),
  },
  {
    path: "/wine_lore",
    name: "The Wine Quality Dataset",
    // route level code-splitting
    // this generates a separate chunk (about.[hash].js) for this route
    // which is lazy-loaded when the route is visited.
    component: () =>
      import(/* webpackChunkName: "wine_lore" */ "../views/Wine.vue"),
  },
  {
    path: "/german_credit_lore",
    name: "The Statlog (German Credit Data) Dataset",
    // route level code-splitting
    // this generates a separate chunk (about.[hash].js) for this route
    // which is lazy-loaded when the route is visited.
    component: () =>
      import(
        /* webpackChunkName: "german_credit_lore" */ "../views/German_credit.vue"
      ),
  },
];

const router = new VueRouter({
  mode: "history",
  base: process.env.BASE_URL,
  routes,
});

export default router;

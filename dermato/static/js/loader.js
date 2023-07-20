function showLoader() {
  const image = document.getElementById("image");
  if (image.value != null && image.value != "" && image.value != undefined) {
    console.log("show loader");
    const loader = document.querySelector("#loader");
    loader.classList.add("loader");
  }
}

function hideLoader() {
  const loader = document.querySelector("#loader");
  loader.classList.remove("loader");
}

document.addEventListener("DOMContentLoaded", function () {
  // Create lightbox overlay
  var overlay = document.createElement("div");
  overlay.className = "lightbox-overlay";
  overlay.innerHTML = '<button class="lightbox-close" aria-label="Close">&times;</button><img>';
  document.body.appendChild(overlay);

  var overlayImg = overlay.querySelector("img");

  // Close lightbox
  function close() {
    overlay.classList.remove("active");
  }
  overlay.addEventListener("click", close);
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") close();
  });

  // Attach click to content images (not blog card images)
  document.addEventListener("click", function (e) {
    var img = e.target;
    if (
      img.tagName === "IMG" &&
      img.closest(".md-content") &&
      !img.closest(".blog-card")
    ) {
      e.preventDefault();
      overlayImg.src = img.src;
      overlayImg.alt = img.alt;
      overlay.classList.add("active");
    }
  });
});

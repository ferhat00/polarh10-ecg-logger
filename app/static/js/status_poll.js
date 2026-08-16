/* Polls the session status endpoint while processing runs; reloads on any
   terminal state. Vanilla JS, no dependencies, fully offline. */
(function () {
  var card = document.getElementById("processing-card");
  if (!card) return;
  var url = card.getAttribute("data-status-url");
  var terminal = ["done", "error", "needs_mapping"];

  function poll() {
    fetch(url)
      .then(function (r) { return r.json(); })
      .then(function (data) {
        if (terminal.indexOf(data.status) !== -1) {
          window.location.reload();
        } else {
          setTimeout(poll, 1500);
        }
      })
      .catch(function () { setTimeout(poll, 3000); });
  }
  setTimeout(poll, 1500);
})();

function closeDialog(dialog) {
  if (dialog?.open) dialog.close();
}

function openDialog(dialog) {
  if (!dialog || dialog.open) return;
  document.querySelectorAll("dialog[open]").forEach(closeDialog);
  dialog.showModal();
}

export function initializeDialogs() {
  document.addEventListener("click", (event) => {
    const openButton = event.target.closest("[data-open-dialog]");
    if (openButton) {
      openDialog(document.getElementById(openButton.dataset.openDialog));
      return;
    }

    const closeButton = event.target.closest("[data-close-dialog]");
    if (closeButton) {
      closeDialog(closeButton.closest("dialog"));
      return;
    }

  });

  document.querySelectorAll("dialog").forEach((dialog) => {
    dialog.addEventListener("click", (event) => {
      if (event.target === dialog) closeDialog(dialog);
    });
  });
}

/**
 * Modal stack - one Escape handler for every hand-rolled modal.
 *
 * Each modal used to add its own document-level `keydown` listener, so a
 * single Escape press closed every modal stacked on screen at once. Modals now
 * register their close callback here when they open and release it when they
 * close; Escape only ever closes the most recently opened one.
 */

/** @type {Array<{close: () => void}>} */
const stack = [];
let listening = false;

function onKeydown(event) {
    if (event.key !== 'Escape' || stack.length === 0) return;
    const top = stack[stack.length - 1];
    event.preventDefault();
    event.stopPropagation();
    top.close();
}

export const ModalStack = {
    /**
     * Register a modal as the top of the stack.
     * @param {() => void} close - Called when Escape is pressed while this
     *   modal is on top. It should close the modal (and normally release it).
     * @returns {() => void} release - Remove this modal from the stack. Safe to
     *   call more than once.
     */
    push(close) {
        if (!listening) {
            document.addEventListener('keydown', onKeydown);
            listening = true;
        }
        const entry = { close };
        stack.push(entry);
        return () => {
            const idx = stack.indexOf(entry);
            if (idx !== -1) stack.splice(idx, 1);
        };
    },
};

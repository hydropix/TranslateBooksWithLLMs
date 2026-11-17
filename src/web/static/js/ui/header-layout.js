/**
 * Header layout switch
 *
 * The top tabs must always stay on a single row. When the logo, the tabs
 * and the status section (LLM status, UI language, theme) no longer fit
 * side by side, the header switches to its stacked layout: logo and status
 * on the first row, tabs alone on a full-width row below (the same layout
 * the mobile breakpoint uses).
 *
 * A fixed breakpoint cannot decide this: tab labels change length with the
 * UI language and the LLM status text changes with the connected model, so
 * the fit is measured instead.
 */

const STACKED_CLASS = 'header-stacked';

function px(value) {
    const n = parseFloat(value);
    return Number.isFinite(n) ? n : 0;
}

function initHeaderLayout() {
    const header = document.querySelector('.header');
    const logo = header?.querySelector('.logo-section');
    const nav = document.getElementById('topTabNav');
    const status = header?.querySelector('.status-section');
    if (!header || !logo || !nav || !status) return;

    const buttons = Array.from(nav.querySelectorAll('.tab-btn'));

    function tabsWidth() {
        const gap = px(getComputedStyle(nav).columnGap);
        const visible = buttons.filter((btn) => btn.offsetParent !== null);
        const sum = visible.reduce((acc, btn) => acc + btn.getBoundingClientRect().width, 0);
        return sum + gap * Math.max(visible.length - 1, 0);
    }

    function update() {
        const headerStyle = getComputedStyle(header);
        const available = header.clientWidth
            - px(headerStyle.paddingLeft)
            - px(headerStyle.paddingRight);
        const gap = px(headerStyle.columnGap);
        const needed = logo.getBoundingClientRect().width
            + tabsWidth()
            + status.getBoundingClientRect().width
            + gap * 2;
        // 1px of slack absorbs sub-pixel rounding so the layout cannot flicker.
        header.classList.toggle(STACKED_CLASS, needed > available + 1);
    }

    let frame = 0;
    const schedule = () => {
        if (frame) return;
        frame = requestAnimationFrame(() => {
            frame = 0;
            update();
        });
    };

    // Header width (window resize), tab label widths (locale switch) and the
    // status width (LLM status text) all change the fit.
    const observer = new ResizeObserver(schedule);
    observer.observe(header);
    observer.observe(logo);
    observer.observe(status);
    buttons.forEach((btn) => observer.observe(btn));

    update();
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initHeaderLayout);
} else {
    initHeaderLayout();
}

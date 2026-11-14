/**
 * File-type icons shared by the file queue, the progress title, the
 * completion card and the history list.
 *
 * Two EPUB placeholder variants exist on purpose and are kept distinct:
 *   - the default one (48x72, inline-sized) used by the progress title, the
 *     completion card and the history list;
 *   - the compact one (viewBox 48x64, sized by the `.generic-epub-icon` CSS
 *     class) used by the file queue list.
 */

/**
 * Generic EPUB cover icon (used when no real thumbnail is available).
 * @returns {string} SVG HTML string
 */
export function createGenericEPUBIcon() {
    return `
            <svg style="width: 48px; height: 72px;" viewBox="0 0 48 72" xmlns="http://www.w3.org/2000/svg">
                <!-- Book cover -->
                <rect x="6" y="3" width="36" height="66" rx="2.5"
                      fill="#5a8ee8" stroke="#3676d8" stroke-width="2"/>
                <!-- Book spine line -->
                <path d="M6 13 L42 13" stroke="#3676d8" stroke-width="1.8"/>
                <!-- Text lines -->
                <path d="M10 22 L38 22 M10 32 L38 32 M10 42 L32 42"
                      stroke="white" stroke-width="2.2" stroke-linecap="round" opacity="0.8"/>
                <!-- EPUB badge -->
                <circle cx="24" cy="56" r="5" fill="white" opacity="0.9"/>
                <text x="24" y="60" text-anchor="middle" font-size="6"
                      fill="#3676d8" font-weight="bold">E</text>
            </svg>
        `;
}

/**
 * Compact generic EPUB icon for the file queue list. Sized by the
 * `.generic-epub-icon` class in style.css rather than inline.
 * @returns {string} SVG HTML string
 */
export function createCompactEPUBIcon() {
    return `
            <svg class="generic-epub-icon" viewBox="0 0 48 64" xmlns="http://www.w3.org/2000/svg">
                <!-- Book cover -->
                <rect x="8" y="4" width="32" height="56" rx="2"
                      fill="#5a8ee8" stroke="#3676d8" stroke-width="2"/>
                <!-- Book spine line -->
                <path d="M8 12 L40 12" stroke="#3676d8" stroke-width="1.5"/>
                <!-- Text lines -->
                <path d="M12 20 L36 20 M12 28 L36 28 M12 36 L30 36"
                      stroke="white" stroke-width="2" stroke-linecap="round" opacity="0.8"/>
                <!-- EPUB badge -->
                <circle cx="24" cy="48" r="4" fill="white" opacity="0.9"/>
                <text x="24" y="51" text-anchor="middle" font-size="5"
                      fill="#3676d8" font-weight="bold">E</text>
            </svg>
        `;
}

/**
 * Generic icon for a file type.
 * @param {string} fileType - File type ('txt', 'epub', 'srt', ...)
 * @param {Object} [options]
 * @param {boolean} [options.compact=false] - Use the compact EPUB variant
 *   (file queue list) instead of the default one.
 * @returns {string} HTML string for the icon
 */
export function getFileIcon(fileType, { compact = false } = {}) {
    if (fileType === 'epub') {
        return compact ? createCompactEPUBIcon() : createGenericEPUBIcon();
    } else if (fileType === 'srt') {
        return '🎬';
    }
    return '📄';
}

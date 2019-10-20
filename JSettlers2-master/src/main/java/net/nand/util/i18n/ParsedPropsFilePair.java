/*
 * nand.net i18n utilities for Java: Property file editor for translators (side-by-side source and destination languages).
 * This file Copyright (C) 2013,2016 Jeremy D Monin <jeremy@nand.net>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 3
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * The maintainer of this program can be reached at jeremy@nand.net
 **/
package net.nand.util.i18n;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;

import net.nand.util.i18n.PropsFileParser.KeyPairLine;

/**
 * Represents a source-language and destination-language pair of properties files.
 * There should be no keys in the target that aren't in the source.
 * Each key in the source/destination is a {@link ParsedPropsFilePair.FileEntry} here.
 *<P>
 * Remember that {@code .properties} bundle files are encoded not in {@code UTF-8} but in {@code ISO-8859-1}:
 *<UL>
 * <LI> <A href="http://docs.oracle.com/javase/1.5.0/docs/api/java/util/Properties.html#encoding"
 *       >java.util.Properties</A> (Java 1.5)
 * <LI> <A href="http://stackoverflow.com/questions/4659929/how-to-use-utf-8-in-resource-properties-with-resourcebundle"
 *       >Stack Overflow: How to use UTF-8 in resource properties with ResourceBundle</A> (asked on 2011-01-11)
 *</UL>
 * Characters outside that encoding must use <code>&#92;uXXXX</code> code escapes.
 *<P>
 * The parsed merged list is {@link #getContents()}, any keys only in the destination are in {@link #getDestOnly()}.
 * The source or destination "half" can be generated with {@link #extractContentsHalf(boolean)}.
 * Changes to that "half" can then be saved with {@link PropsFileWriter}.
 *
 * @author Jeremy D Monin &lt;jeremy@nand.net&gt;
 */
public class ParsedPropsFilePair
{
    /** If true, there are unsaved changes from editing the pair.
     *  If new rows were inserted, {@link #unsavedInsRows} will also be set.
     */
    public boolean unsavedSrc, unsavedDest;

    /** If true, there are unsaved inserted rows.
     *  You can call {@link #convertInsertedRows()} to make them key or comment rows.
     *  To avoid inconsistencies when saving changes, {@link #extractContentsHalf(boolean)}
     *  will call that method if {@code unsavedInsRows}.
     *  If you call {@link #convertInsertedRows()} yourself first, you can check its return value
     *  to see if any conversions were needed.
     */
    public boolean unsavedInsRows;

    public final File srcFile, destFile;

    /** If true, {@link #setDestIsNew(List)} has been called. */
    private boolean isDestNew;

    /** Logical entries, one per key, to be expanded into {@link #cont}:
     *  Source and dest file-pair key-by-key contents, from parsing; does not contain {@link #destOnlyPairs}.
     *  Built during {@link #parseSrc()}, updated during {@link #parseDest()}.
     *  @see #srcParsedDupeKeys
     *  @see #destParsedDupeKeys
     */
    private List<FileKeyEntry> parsed;

    /**
     * Any duplicate keys found during parsing, or {@code null} if none.
     * Key = each key seen more than once while parsing, value = values for that key.
     * Built during {@link #parseSrc()} or {@link #parseDest()}.
     *<P>
     * For structure details see {@link PropsFileParser#findDuplicateKeys(List, Map)},
     * including special case of 'duplicates' with same value.
     * @see #parsed
     * @see #cont
     */
    private Map<String, String> srcParsedDupeKeys, destParsedDupeKeys;

    /** Expanded entries (contents), one per line in file, from {@link #parsed}:
     *  Source and dest file-pair line-by-line grid contents, from parsing and editing;
     *  also contains {@link #destOnlyPairs}.  Built during {@link #parseDest()} or {@link #setDestIsNew(List)}
     *  by {@link #buildContFromSrcDest(Map)}.
     *  @see #srcParsedDupeKeys
     *  @see #destParsedDupeKeys
     */
    private List<FileEntry> cont;

    /** If the file starts with a comment followed by blank lines above the first a key-value pair, comment goes in {@link #cont} and also goes here */
    private FileKeyEntry contHeadingComment;

    /** If the file ends with a comment not followed by a key-value pair, it goes in {@link #cont} and also goes here */
    private FileKeyEntry contEndingComment;

    /**
     * Key-value pairs found only in the destination file, not the source file; or {@code null} if none.
     * Initialized in {@link #parseDest()} if needed.
     */
    private List<PropsFileParser.KeyPairLine> destOnlyPairs;

    /**
     * Create a new empty FilePair to begin parsing.
     *<P>
     * If {@code src} and {@code dest} already exist on disk,
     * call {@link #parseSrc()} and then {@link #parseDest()}
     * to read them into this object.
     *
     * @param src  Source language/locale's properties file
     * @param dest  Destination language/locale's properties file
     */
    public ParsedPropsFilePair(final File src, final File dest)
    {
        srcFile = src;
        destFile = dest;
        parsed = new ArrayList<FileKeyEntry>();
        cont = new ArrayList<FileEntry>();
    }

    /**
     * Get the number of key-value pairs in {@link #getContents()}.
     * Each row can be retrieved with {@link #getRow(int)}.
     */
    public int size() { return cont.size(); }

    /**
     * Get the list of key-value pairs found in the source and maybe also the destination.
     * The size of this list is {@link #size()}.
     * For access to individual rows, use {@link #getRow(int)}.
     * @see #extractContentsHalf(boolean)
     * @see #getSrcDupeKeys()
     * @see #getDestDupeKeys()
     */
    public Iterator<FileEntry> getContents() { return cont.iterator(); }

    /**
     * Get the source or destination "half" of the parsed pair's contents,
     * in a format suitable for {@link PropsFileWriter#write(List, String)}.
     *<P>
     * To avoid inconsistencies when saving changes, checks {@link #unsavedInsRows}
     * and will call {@link #convertInsertedRows()} if needed.
     *<P>
     * Lines with a key but no value are ignored and not extracted.
     *
     * @param destHalf  True for destination half, false for source half
     * @return  List of key pair lines; may contain comment lines (null keys).
     * @see #getContents()
     */
    public List<KeyPairLine> extractContentsHalf(final boolean destHalf)
    {
        if (unsavedInsRows)
            convertInsertedRows();

        ArrayList<KeyPairLine> ret = new ArrayList<KeyPairLine>(cont.size());
        for (final FileEntry kp: cont)
        {
            if (kp instanceof FileCommentEntry)
            {
                String commentLine;
                if (destHalf)
                    commentLine = ((FileCommentEntry) kp).destComment;
                else
                    commentLine = ((FileCommentEntry) kp).srcComment;

                if (commentLine == null)
                    ret.add(new KeyPairLine(null));
                else {
                    ArrayList<String> oneline = new ArrayList<String>(2);
                    oneline.add(commentLine);
                    ret.add(new KeyPairLine(oneline));
                }
            } else {
                FileKeyEntry kpe = (FileKeyEntry) kp;
                final CharSequence val;
                final boolean spc;
                if (destHalf)
                {
                    val = kpe.destValue;
                    spc = kpe.destSpacedEquals;
                } else {
                    val = kpe.srcValue;
                    spc = kpe.srcSpacedEquals;
                }

                if (val == null)
                    continue;  // skip it: only the other half of the dest/src pair has a value for this key

                ret.add(new KeyPairLine
                        (kpe.key, val.toString(), null, spc));
            }
        }

        return ret;
    }

    /**
     * Get row {@code r} of the contents.
     * This is a reference, not a copy; if you change its fields,
     * be sure to set the {@link #unsavedDest} and/or {@link #unsavedSrc} flags.
     * @param r  A row number within the contents, 0 &lt;= {@code r} &lt; {@link #size()}
     * @return  The FileEntry in row {@code r}
     * @throws ArrayIndexOutOfBoundsException
     */
    public FileEntry getRow(final int r)
        throws ArrayIndexOutOfBoundsException
    {
        if ((r < 0) || (r >= cont.size()))
            throw new ArrayIndexOutOfBoundsException(r);
        return cont.get(r);
    }

    /**
     * Is this key found only in the destination file, not in the source file?
     * This is an error and a rare occurrence.
     */
    public boolean isKeyDestOnly(final String key)
    {
        if (destOnlyPairs == null)
            return false;

        // rare, so 0 or not many keys; linear search is OK
        for (PropsFileParser.KeyPairLine kp : destOnlyPairs)
            if (key.equals(kp.key))
                return true;

        return false;
    }

    /**
     * Get the number of key-value pairs found only in the destination, or 0,  in {@link #getDestOnly()}.
     * @see #isKeyDestOnly(String)
     */
    public int getDestOnlySize()
    {
        return (destOnlyPairs != null) ? destOnlyPairs.size() : 0;
    }

    /**
     * Get the list of key-value pairs found only in the destination, or {@code null} if none.
     * The size of this list is {@link #getDestOnlySize()}.
     * @see #getContents()
     * @see #getDestDupeKeys()
     */
    public Iterator<PropsFileParser.KeyPairLine> getDestOnly()
    {
        return (destOnlyPairs != null) ? destOnlyPairs.iterator() : null;
    }

    /**
     * Get any keys seen during source parsing more than once with different values.
     * Same map format as {@link PropsFileParser#findDuplicateKeys(List, Map)}.
     * @return  Duplicate keys in source file, or {@code null} if none
     * @see #getDestDupeKeys()
     * @see #getContents()
     * @see #getDestOnly()
     */
    public Map<String, String> getSrcDupeKeys()
    {
        return srcParsedDupeKeys;
    }

    /**
     * Get any keys seen during destination parsing more than once with different values.
     * Same map format as {@link PropsFileParser#findDuplicateKeys(List, Map)}.
     * @return  Duplicate keys in destination file, or {@code null} if none
     * @see #getSrcDupeKeys()
     * @see #getContents()
     */
    public Map<String, String> getDestDupeKeys()
    {
        return destParsedDupeKeys;
    }

    /**
     * Parse the source-language file at {@link #srcFile}.
     * After calling this method, call {@link #parseDest()} or {@link #setDestIsNew(List)}.
     * @throws IllegalStateException  if we've already read or created entries in this object; {@link #size()} != 0
     * @throws IOException  if file not found, cannot be read, etc
     */
    public void parseSrc()
        throws IllegalStateException, IOException
    {
        if (! parsed.isEmpty())
            throw new IllegalStateException("cannot call parseSrc unless object is empty");

        Map<String, String> dupeKeys = new HashMap<String, String>();
        final List<PropsFileParser.KeyPairLine> srcLines = PropsFileParser.parseOneFile(srcFile, dupeKeys);
        if (srcLines.isEmpty())
            return;
        if (! dupeKeys.isEmpty())
            srcParsedDupeKeys = dupeKeys;

        final PropsFileParser.KeyPairLine firstLine = srcLines.get(0);
        if ((firstLine.key == null) && (firstLine.comment != null))
        {
            FileKeyEntry fe = new FileKeyEntry(firstLine.comment);
            parsed.add(fe);
            contHeadingComment = fe;
            srcLines.remove(0);   // main loop expects line.key except for very last entry
        }

        for (final PropsFileParser.KeyPairLine line : srcLines)
        {
            if (line.key != null)
            {
                FileKeyEntry fe = new FileKeyEntry(line.key, line.value);
                if (line.comment != null)
                    fe.srcComment = line.comment;
                fe.srcSpacedEquals = line.spacedEquals;
                parsed.add(fe);
            } else if (line.comment != null) {
                FileKeyEntry fe = new FileKeyEntry(line.comment);
                parsed.add(fe);
                if (contEndingComment != null)
                    throw new IllegalStateException("src: file-ending comment already exists");
                contEndingComment = fe;
            }
        }
    }

    /**
     * Call this method to indicate that {@link #destFile} is new and does not yet exist.
     * Optionally provide header comments to place in the destination.
     *<P>
     * Call {@link #parseSrc()} before calling this method.
     * If you call this method, do not call {@link #parseDest()}.
     *
     * @param comments  any comment lines to use as the initial contents;
     *     same format as {@link PropsFileParser.KeyPairLine#comment}. Otherwise {@code null}.
     * @throws IllegalStateException  if {@code parseSrc()} wasn't called yet, or src was empty; {@link #size()} == 0
     */
    public void setDestIsNew(List<String> comments)
        throws IllegalStateException
    {
        if (parsed.isEmpty())
            throw new IllegalStateException("call parseSrc first");

        isDestNew = true;

        if (contHeadingComment == null)
        {
            FileKeyEntry fe = new FileKeyEntry(comments);
            fe.destComment = fe.srcComment;  // constructor sets srcComment, we need destComment instead
            fe.srcComment = null;
            contEndingComment = fe;
        } else {
            contHeadingComment.destComment = comments;
        }
        parsed.get(0).destComment = comments;

        buildContFromSrcDest(null);
        if (contEndingComment != null)
            expandCommentLinesIntoCont(contEndingComment);
    }

    /**
     * Parse the destination-language file at {@link #destFile};
     * call {@link #parseSrc()} before calling this method, so this method
     * can merge the structures together into {@link #cont}.
     * Do not call this method if you have called {@link #setDestIsNew(List)}.
     *<P>
     * Merging is done using {@link #parsed}, {@link #contHeadingComment}, and {@link #contEndingComment}.
     * Any destination keys not found in source (in {@code #parsed}) are placed into {@link #destOnlyPairs}.
     *
     * @throws IllegalStateException  if {@code parseSrc()} wasn't called yet, or src was empty; {@link #size()} == 0;
     *     or if {@link #setDestIsNew(List)} has been called
     * @throws IOException  if file not found, cannot be read, etc
     */
    public void parseDest()
        throws IllegalStateException, IOException
    {
        if (parsed.isEmpty())
            throw new IllegalStateException("call parseSrc first");
        if (isDestNew)
            throw new IllegalStateException("do not call both parseDest and setDestIsNew");

        Map<String, String> dupeKeys = new HashMap<String, String>();
        final List<PropsFileParser.KeyPairLine> destLines = PropsFileParser.parseOneFile(destFile, dupeKeys);
        if (destLines.isEmpty())
            return;
        if (! dupeKeys.isEmpty())
            destParsedDupeKeys = dupeKeys;

        final PropsFileParser.KeyPairLine firstLine = destLines.get(0);
        if ((firstLine.key == null) && (firstLine.comment != null))
        {
            if (contHeadingComment == null)
            {
                FileKeyEntry fe = new FileKeyEntry(firstLine.comment);
                fe.destComment = fe.srcComment;  // constructor sets srcComment, we need destComment instead
                fe.srcComment = null;
                contHeadingComment = fe;
            } else {
                contHeadingComment.destComment = firstLine.comment;
            }
            destLines.remove(0);   // upcoming destLines loop expects line.key != null, except for very last entry
        }

        /** Map from destination keys in destLines to {@link #cont} entries.
         * Does not contain {@link #contHeadingComment} or {@link #contEndingComment} because their key would be null.
         */
        Map<String, PropsFileParser.KeyPairLine> destKeys = new HashMap<String, PropsFileParser.KeyPairLine>();

        // Go through destLines first, to build destKeys and set contEndingComment.
        // Remove the ending comment if any from destLines.
        for (final PropsFileParser.KeyPairLine line : destLines)
        {
            if (line.key != null)
            {
                destKeys.put(line.key, line);
            } else if (line.comment != null) {
                // key is null, comment isn't: must be the comment(s) after the last key in the file
                if (contEndingComment == null)
                {
                    FileKeyEntry fe = new FileKeyEntry(line.comment);
                    fe.destComment = fe.srcComment;  // constructor sets srcComment, we need destComment instead
                    fe.srcComment = null;
                    contEndingComment = fe;
                } else {
                    if (contEndingComment.destComment != null)
                        throw new IllegalStateException("dest: file-ending comment already exists");
                    contEndingComment.destComment = line.comment;
                }
            }
        }
        if ((contEndingComment != null) && (contEndingComment.destComment != null))
            destLines.remove(destLines.size() - 1);

        // Loop through srcLines to build the final content list
        // in the same order as the source file.
        // Keys found in both src and dest will have their
        // destKeys entry's key field changed to null.
        buildContFromSrcDest(destKeys);

        // Build and add destOnlyPairs here, from destLines where key still is != null
        for (final PropsFileParser.KeyPairLine line : destLines)
        {
            if (line.key == null)
                continue;

            if (destOnlyPairs == null)
                destOnlyPairs = new ArrayList<PropsFileParser.KeyPairLine>();
            destOnlyPairs.add(line);

            FileKeyEntry fe = new FileKeyEntry(line.key, line.value);
            fe.destValue = fe.srcValue;  // constructor sets srcValue, we need destValue instead
            fe.srcValue = null;
            if (line.comment != null)
                fe.destComment = line.comment;
            fe.destSpacedEquals = line.spacedEquals;
            cont.add(fe);
        }

        // Finally, the file-ending comment, if any
        if (contEndingComment != null)
            expandCommentLinesIntoCont(contEndingComment);
    }

    /**
     * Build {@link #cont} by combining the parsed source ({@link #parsed}) and destination ({@code destKeys}).
     * Called from {@link #parseDest()} or {@link #setDestIsNew(List)}.
     * Expands {@link #contHeadingComment} before iterating through keys.
     *<P>
     * Because {@link #parseDest()} adds entries to {@link #cont} after calling this method
     * (for {@link #destOnlyPairs}), this method doesn't expand {@link #contEndingComment};
     * the caller must do so by calling {@link #expandCommentLinesIntoCont(FileKeyEntry) expandCommentLinesIntoCont}
     * ({@link #contEndingComment}).
     *
     * @param destKeys  Map from destination keys in destLines to {@link #cont} entries, or {@code null}.
     *     Does not contain {@link #contHeadingComment} or {@link #contEndingComment} because their key would be null.
     *     <P>
     *     <B>Note:</B> {@code destKeys} contents are modified by this method:<BR>
     *     As entries in this map are matched to source entries with the same key, each matched entry's
     *     {@link PropsFileParser.KeyPairLine#key KeyPairLine.key} field will be set to {@code null}.
     *     If no source line has the same key as a given {@code destKeys} entry, at the end of the method
     *     that entry's key field will still be {@code != null}.
     */
    private void buildContFromSrcDest(final Map<String, PropsFileParser.KeyPairLine> destKeys)
    {
        if (contHeadingComment != null)
            expandCommentLinesIntoCont(contHeadingComment);

        for (final FileKeyEntry srcLine : parsed)
        {
            if (srcLine.key == null)
                continue;  // ignore comments at start or end of file

            PropsFileParser.KeyPairLine dest = (destKeys != null) ? destKeys.get(srcLine.key) : null;
            if (dest != null)
            {
                srcLine.destValue = dest.value;
                if (dest.comment != null)
                    srcLine.destComment = dest.comment;
                srcLine.destSpacedEquals = dest.spacedEquals;

                if ((srcLine.srcComment != null) || (srcLine.destComment != null))
                    expandCommentLinesIntoCont(srcLine);
                cont.add(srcLine);

                dest.key = null;  // mark as matched to src, not left over for destOnlyPairs in parseDest()
            } else {
                // this key is in source only
                if (srcLine.srcComment != null)
                    expandCommentLinesIntoCont(srcLine);
                cont.add(srcLine);
            }
        }
    }

    /** Expand this key-value entry's preceding comment(s) to new lines appended at the end of {@link #cont}. */
    private final void expandCommentLinesIntoCont(final FileKeyEntry fe)
    {
        // TODO line up comments on lines above src, dest: bottom-justify, not top-justify, if not the same number of comment lines
        final int nSrc  = (fe.srcComment != null)  ? fe.srcComment.size()  : 0,
                  nDest = (fe.destComment != null) ? fe.destComment.size() : 0,
                  nComment = (nSrc >= nDest) ? nSrc : nDest;
        for (int i = 0; i < nComment; ++i)
            cont.add(new FileCommentEntry
                    ( ((i < nSrc)  ? fe.srcComment.get(i)  : null),
                      ((i < nDest) ? fe.destComment.get(i) : null) ));
    }

    /**
     * Add/insert a row before or after an existing row.
     * Added rows are {@link FileKeyEntry} until {@link #convertInsertedRows()} is called,
     * they may become {@link FileCommentEntry} at that time.
     *
     * @param r  Row number
     * @param beforeRow  If true, insert before (above), otherwise add after (below) this line; ignored if
     *     adding at end ({@code r} == {@link #size()})
     * @throws IndexOutOfBoundsException  if {@code r} is &lt; 0 or &gt; {@link #size()}
     */
    public void insertRow(int r, final boolean beforeRow)
        throws IndexOutOfBoundsException
    {
        if (! beforeRow)
            ++r;

        FileKeyEntry fke = new FileKeyEntry(null, null);
        fke.newAdd = true;
        if (r < cont.size())
            cont.add(r, fke);  // throws IndexOutOfBoundsException if out of range
        else
            cont.add(fke);

        if (! unsavedInsRows)
            unsavedInsRows = true;
    }

    /**
     * Check for rows added by the editor, with the {@link FileKeyEntry#newAdd} flag;
     * inspect these for keys and values, and if needed convert them to {@link FileCommentEntry}.
     * @return  True if any rows had the flag and had their contents converted (keys, values, or comments)
     */
    public boolean convertInsertedRows()
    {
        boolean foundAny = false;

        for (ListIterator<FileEntry> li = cont.listIterator(); li.hasNext(); )
        {
            FileEntry fe = li.next();
            if (! (fe instanceof FileKeyEntry))
                continue;

            FileKeyEntry fke = (FileKeyEntry) fe;
            if (! fke.newAdd)
                continue;

            if ((fke.key == null) || (fke.key.length() == 0))
            {
                // No key: this is a comment or blank line
                foundAny = true;

                String srcComm  = (fke.srcValue != null)  ? fke.srcValue.toString().trim()  : null,
                       destComm = (fke.destValue != null) ? fke.destValue.toString().trim() : null;
                if ((srcComm != null) && ! srcComm.startsWith("#"))
                    srcComm = "# " + srcComm;
                if ((destComm != null) && ! destComm.startsWith("#"))
                    destComm = "# " + destComm;

                FileCommentEntry fce = new FileCommentEntry(srcComm, destComm);
                li.set(fce);
            }
        }

        unsavedInsRows = false;
        return foundAny;
    }

    //
    // Nested Classes
    //

    /**
     * One message key or comment line, with its source and destination languages' string values and comments.
     * @see FileKeyEntry
     * @see FileCommentEntry
     */
    public static abstract class FileEntry {}

    /** Comment line in source and destination */
    public static final class FileCommentEntry extends FileEntry
    {
        /** Comment field strings start with {@code #} and are trimmed; may be null or "" on blank lines. */
        public String srcComment, destComment;

        public FileCommentEntry(final String srcComment)
        {
            this.srcComment = srcComment;
        }

        public FileCommentEntry(final String srcComment, final String destComment)
        {
            this.srcComment = srcComment;
            this.destComment = destComment;
        }

        public final String toString() { return "FileCommentEntry"; }
    }

    /**
     * Key-value pair line in source and destination, with preceding comments, or multi-line
     * comment at the end of the file.
     *<P>
     * Also used by editor for newly inserted lines, when we are unsure if they will be comments or key-value.
     * To convert those before saving, call {@link ParsedPropsFilePair#convertInsertedRows()}.
     */
    public static final class FileKeyEntry extends FileEntry
    {
        /**
         * Is this line newly added in the editor during this edit session?
         * Remains true even after saving the file, so its {@link #key} can still be edited.
         * @see ParsedPropsFilePair#convertInsertedRows()
         */
        public boolean newAdd;

        /** key for retrieval, or {@code null} for comments at the end of the file */
        public String key;

        /** Preceding comment lines and/or blank lines in the source language, or {@code null};
         *  same format as {@link PropsFileParser.KeyPairLine#comment}
         */
        public List<String> srcComment;

        /** Preceding comment lines and/or blank lines in the destination language, or {@code null};
         *  same format as {@link PropsFileParser.KeyPairLine#comment}
         */
        public List<String> destComment;

        /** Value in source language, or {@code null} for a key defined only in the destination file, or for
         *  empty strings or only whitespace.
         */
        public CharSequence srcValue;

        /** Value in destination language, or {@code null}. Empty strings or only whitespace are {@code null}. */
        public CharSequence destValue;

        /** If true, the key and value are separated by " = " instead of "=".
         *  This is tracked to minimize whitespace changes when editing a properties file.
         *  True by default except comment lines; set false after constructor if needed.
         */
        public boolean srcSpacedEquals, destSpacedEquals;

        /**
         * Create an entry for a source language (no destination value yet).
         * @param key
         * @param srcValue
         * @throws IllegalArgumentException if {@code key} is null
         */
        public FileKeyEntry(final String key, final String srcValue)
            throws IllegalArgumentException
        {
            this(key, srcValue, null);
        }

        /**
         * Create an entry for a source and destination.
         * @param key  Key for retrieval.  May be {@code null} only if inserting a new line which may become a comment.
         *                Before saving, set the key and value, or convert to {@link FileCommentEntry}.
         * @param srcValue  Value in source language, or {@code null} for empty or only whitespace
         * @param destValue  Value in destination language, or {@code null} for undefined, empty or only whitespace
         * @throws IllegalArgumentException if {@code key} is null
         */
        public FileKeyEntry(final String key, final String srcValue, final String destValue)
            throws IllegalArgumentException
        {
            this.key = key;
            this.srcValue = srcValue;
            this.destValue = destValue;
            srcSpacedEquals = true;
            destSpacedEquals = true;
        }

        /**
         * Create an entry that's only a comment without a following key-value pair (at the end of the file).
         * @param srcComment  Comment text in the source language file
         */
        public FileKeyEntry(final List<String> srcComment)
        {
            this.key = null;
            this.srcComment = (srcComment.isEmpty()) ? null : srcComment;
        }

        /** toString includes the source and destination values, and number of source and destination comment lines. */
        public final String toString()
        {
            return "{key=" + key
                    + ((srcValue != null) ? (" src=#" + srcValue + "#") : " src=null")
                    + ((destValue != null) ? (" dest=#" + destValue + "#") : " dest=null")
                    + ((srcComment != null) ? (" srcComment=" + srcComment.size()) : "")
                    + ((destComment != null) ? (" destComment=" + destComment.size()) : "")
                    + "}";
        }

    }

}

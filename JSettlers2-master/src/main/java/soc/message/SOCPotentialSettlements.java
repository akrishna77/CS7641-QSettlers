/**
 * Java Settlers - An online multiplayer version of the game Settlers of Catan
 * Copyright (C) 2003  Robert S. Thomas <thomas@infolab.northwestern.edu>
 * Portions of this file Copyright (C) 2010-2014,2017-2019 Jeremy D Monin <jeremy@nand.net>
 * Portions of this file Copyright (C) 2012 Paul Bilnoski <paul@bilnoski.net>
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
 * The maintainer of this program can be reached at jsettlers@nand.net
 **/
package soc.message;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.StringTokenizer;

/**
 * This message contains a list of potential settlements.
 *<P>
 * In version 2.0.00 and newer:
 *<UL>
 *<LI> This message is sent before any {@link SOCPutPiece}.
 *   So even if game is in progress, each player receives their unique potential settlement node list,
 *   to populate their legal node/edge sets if using {@link soc.game.SOCBoardLarge SOCBoardLarge},
 *   before seeing any of their piece locations.
 *<LI> <tt>playerNumber</tt> can be -1
 *   to indicate this message applies to all players.  For the
 *   SOCBoardLarge encoding only, that also indicates
 *   the legal settlements should be set and the
 *   legal roads recalculated from this message's list of potentials.
 *   <P>
 *   If the game has already started, Land Area contents are sent when
 *   <tt>playerNumber</tt> == 0 and board's legal roads should be
 *   calculated at that point, before calculating any player's
 *   legal or potential sets.
 *<LI> More than one "land area" (group of islands, or subset of islands)
 *   can be designated; can also require the player to start
 *   the game in a certain land area ({@link #startingLandArea}).
 *   Each Land Area is sent as a list of that Area's nodes.
 *<LI> Players can build ships on any sea or coastal edge, except in
 *   scenario {@code _SC_PIRI} which restricts them to certain edges;
 *   see optional field {@link #legalSeaEdges}.
 *</UL>
 *<P>
 * In scenario {@code _SC_PIRI}, after initial placement, each player can place
 * not only at these potential locations but also at their "lone settlement"
 * node previously sent in {@link SOCBoardLayout2} as layout part {@code "LS"}.
 *
 * @author Robert S Thomas
 */
public class SOCPotentialSettlements extends SOCMessage
    implements SOCMessageForGame
{
    private static final long serialVersionUID = 2000L;  // last structural change v2.0.00

    /**
     * In version 2.0.00 and above, playerNumber can be -1
     * to indicate all players have these potential settlements.
     * @since 2.0.00
     */
    public static final int VERSION_FOR_PLAYERNUM_ALL = 2000;

    /**
     * Name of game
     */
    private String game;

    /**
     * Player number, or -1 for all players (version 2.0.00 or newer)
     */
    private int playerNumber;

    /**
     * Potential settlement node coordinates for {@link #playerNumber}.
     *<P>
     * Before v2.0.00 this field was {@code psList}.
     * @see #landAreasLegalNodes
     * @see #psNodesFromAll
     */
    private List<Integer> psNodes;

    /**
     * True if {@link #startingLandArea} == 0 and {@link #psNodes} is merely the
     * union of all node sets in {@link #landAreasLegalNodes}.
     *<P>
     * False otherwise, or if not applicable because {@link #areaCount} == 1
     * and {@link #landAreasLegalNodes} == null.
     * @since 2.0.00
     */
    private final boolean psNodesFromAll;

    /**
     * How many land areas are defined on this board.
     * Always 1 in classic boards (before v2.0.00).
     * @since 2.0.00
     */
    public final int areaCount;

    /**
     * Which land area number within {@link #landAreasLegalNodes} contains {@link #psNodes}?
     * 0 if none because the game already started and {@code psNodes} is unique to the player.
     * 0 if none when game is starting now and players can place anywhere.
     *<P>
     * Not used if {@link #areaCount} == 1.
     * @since 2.0.00
     */
    public final int startingLandArea;

    /**
     * Each land area's legal node coordinates.
     * Index 0 is unused, not a Land Area number.
     *<P>
     * Null if {@link #areaCount} == 1.
     * @since 2.0.00
     * @see #startingLandArea
     * @see #psNodes
     */
    public final HashSet<Integer>[] landAreasLegalNodes;

    /**
     * Optional field for legal sea edges per player for ships, if restricted.
     * Usually {@code null}, because all sea edges are legal except in scenario {@code _SC_PIRI}.
     *<P>
     * If {@link #playerNumber} != -1, {@code legalSeaEdges} contains 1 array, the legal sea edges for that player.
     * Otherwise contains 1 array for each player position (total 4 or 6 arrays).
     *<P>
     * Each per-player array is the same format as in {@code SOCBoardAtServer.PIR_ISL_SEA_EDGES}:
     * A list of individual sea edge coordinates and/or ranges.
     * Ranges are designated by a pair of positive,negative numbers:
     * 0xC04, -0xC0D is a range of the valid edges from C04 through C0D inclusive.
     * If a player position is vacant, their subarray may be empty (length 0) or contain a single zero: <tt>{ 0 }</tt>.
     * @since 2.0.00
     */
    public int[][] legalSeaEdges;

    /**
     * Create a SOCPotentialSettlements message for a board layout without land areas.
     *
     * @param ga  name of the game
     * @param pn  the player number, or -1 for all players in v2.0.00
     *   or newer (see <tt>ps</tt> for implications)
     * @param ps  the list of potential settlement nodes; if <tt>pn == -1</tt>
     *   and the client and server are at least
     *   version 2.0.00 ({@link #VERSION_FOR_PLAYERNUM_ALL}),
     *   <tt>ps</tt> also is the list of legal settlements.
     * @see #SOCPotentialSettlements(String, int, int, HashSet[], int[][])
     */
    public SOCPotentialSettlements(String ga, int pn, List<Integer> ps)
    {
        messageType = POTENTIALSETTLEMENTS;
        game = ga;
        playerNumber = pn;
        psNodes = ps;
        psNodesFromAll = false;
        areaCount = 1;
        landAreasLegalNodes = null;
        startingLandArea = 1;
        legalSeaEdges = null;
    }

    /**
     * Create a SOCPotentialSettlements message for a board layout with multiple land areas,
     * each of which have a set of legal settlements, but only one of which
     * has potential settlements at this time.
     *
     * @param ga  name of the game
     * @param pn  the player number, or -1 for all players
     * @param pan  Potential settlements' land area number, or 0 if the game has started and so player's
     *     unique list of potential settlements doesn't match any of the land area coordinate lists.
     *     In that case use <tt>lan[0]</tt> to hold the potential settlements node list.
     *     <P>
     *     If the game is just starting and the player can start anywhere (<tt>pan == 0</tt>),
     *     then <tt>lan[0]</tt> should be <tt>null</tt>.
     *     <P>
     *     From {@link soc.game.SOCBoardLarge#getStartingLandArea()}.
     * @param lan  Each land area's legal node lists.
     *     List at index number <tt>pan</tt> will be sent as the list of potential settlements.
     *     If <tt>pan</tt> is 0 because game has started (see above), use index 0 for the player's
     *     potentials list (which may be empty). Otherwise index 0 is unused (<tt>null</tt>).
     *     <P>
     *     If the game is just starting and the player can start anywhere (<tt>pan == 0</tt>),
     *     then <tt>lan[0]</tt> should be <tt>null</tt> and the {@link #getPotentialSettlements()} list
     *     will be formed by combining <tt>lan[1] .. lan[n-1]</tt>.
     * @param lse  Legal sea edges for ships if restricted, or {@code null}; see {@link #legalSeaEdges} field for format
     * @throws IllegalArgumentException  if <tt>ln[pan] == null</tt> and <tt>pan != 0</tt>,
     *            or if <tt>ln[<i>i</i>]</tt> == <tt>null</tt> for any <i>i</i> &gt; 0
     * @see #SOCPotentialSettlements(String, int, List)
     * @since 2.0.00
     */
    public SOCPotentialSettlements(String ga, int pn, final int pan, HashSet<Integer>[] lan, final int[][] lse)
        throws IllegalArgumentException
    {
        messageType = POTENTIALSETTLEMENTS;
        game = ga;
        playerNumber = pn;
        psNodesFromAll = (pan == 0) && (lan[0] == null);
        if (! psNodesFromAll)
        {
            if (lan[pan] == null)
                throw new IllegalArgumentException();
            psNodes = new ArrayList<Integer>(lan[pan]);
        } else {
            psNodes = new ArrayList<Integer>();
        }
        areaCount = lan.length - 1;
        landAreasLegalNodes = lan;
        startingLandArea = pan;
        legalSeaEdges = lse;

        // consistency-check land areas
        // and if psNodesFromAll, add all nodes to psNodes
        for (int i = 1; i < lan.length; ++i)
        {
            if (lan[i] == null)
                throw new IllegalArgumentException();
            if (psNodesFromAll)
                psNodes.addAll(lan[i]);
        }
    }

    /**
     * @return the game name
     */
    public String getGame()
    {
        return game;
    }

    /**
     * @return the player number
     */
    public int getPlayerNumber()
    {
        return playerNumber;
    }

    /**
     * @return the list of potential settlements
     */
    public List<Integer> getPotentialSettlements()
    {
        return psNodes;
    }

    /**
     * POTENTIALSETTLEMENTS formatted command, for a message with 1 or multiple land areas.
     * Format will be either {@link #toCmd(String, int, List)}
     * or {@link #toCmd(String, int, int, HashSet[], int[][])}.
     *
     * @return the command String
     */
    @Override
    public String toCmd()
    {
        if (landAreasLegalNodes == null)
            return toCmd(game, playerNumber, psNodes);
        else
            return toCmd(game, playerNumber, startingLandArea, landAreasLegalNodes, legalSeaEdges);
    }

    /**
     * <tt>toCmd</tt> for a SOCPotentialSettlements message with 1 land area.
     *<P><tt>
     * POTENTIALSETTLEMENTS sep game sep2 playerNumber sep2 psNodes
     *</tt>
     * @param ga  the game name
     * @param pn  the player number
     * @param ps  the list of potential settlements
     * @return    the command string
     * @see #toCmd(String, int, int, HashSet[], int[][])
     */
    public static String toCmd(String ga, int pn, List<Integer> ps)
    {
        String cmd = POTENTIALSETTLEMENTS + sep + ga + sep2 + pn;

        for (Integer number : ps)
        {
            cmd += (sep2 + number);
        }

        return cmd;
    }

    /**
     * <tt>toCmd</tt> for a SOCPotentialSettlements message with multiple land areas,
     * each of which have a set of legal settlements, but only one of which
     * has potential settlements at this time.
     *<P><pre>
     * POTENTIALSETTLEMENTS sep game sep2 playerNumber sep2 psNodes
     *    sep2 NA sep2 <i>(number of areas)</i> sep2 PAN sep2 <i>(pan)</i>
     *    { sep2 LA<i>#</i> sep2 legalNodesList }+
     *    { sep2 SE { sep2 (legalSeaEdgesList | 0) } }*
     *</pre>
     *<UL>
     * <LI> LA# is the land area number "LA1", "LA2", etc.
     * <LI> None of the LA#s will be <i>(pan)</i> because that list would only repeat
     *      the contents of {@code psNodes}.
     * <LI> If {@code psNodes} is empty (not null) it's sent as the single node {@code 0} which isn't a valid
     *      {@link soc.game.SOCBoardLarge} node coordinate.
     *</UL>
     *
     * @param ga  name of the game
     * @param pn  the player number, or -1 for all players
     * @param pan  Potential settlements' land area number, or 0 if the game has started and so player's
     *     unique list of potential settlements doesn't match any of the land area coordinate lists.
     *     In that case use <tt>lan[0]</tt> to hold the potential settlements node list,
     *     which may be empty.
     *     <P>
     *     From {@link soc.game.SOCBoardLarge#getStartingLandArea()}.
     * @param lan  Each land area's legal node lists.
     *     List at index number <tt>pan</tt> will be sent as the list of potential settlements.
     *     If <tt>pan</tt> is 0 because game has started (see above), use index 0 for the player's
     *     potentials list. Otherwise index 0 is unused (<tt>null</tt>).
     * @param lse  Legal sea edges for ships if restricted, or {@code null}; see {@link #legalSeaEdges} field for format
     * @return   the command string
     * @see #toCmd(String, int, List)
     */
    public static String toCmd(String ga, int pn, final int pan, final HashSet<Integer>[] lan, final int[][] lse)
    {
        StringBuffer cmd = new StringBuffer(POTENTIALSETTLEMENTS + sep + ga + sep2 + pn);

        if (lan[pan] != null)
        {
            if (! lan[pan].isEmpty())
            {
                Iterator<Integer> siter = lan[pan].iterator();
                while (siter.hasNext())
                {
                    int number = siter.next().intValue();
                    cmd.append(sep2);
                    cmd.append(number);
                }
            } else {
                cmd.append(sep2);
                cmd.append(0);
            }
        }

        cmd.append(sep2);
        cmd.append("NA");  // number of areas
        cmd.append(sep2);
        cmd.append(lan.length - 1);

        cmd.append(sep2);
        cmd.append("PAN");
        cmd.append(sep2);
        cmd.append(pan);

        for (int i = 1; i < lan.length; ++i)
        {
            if (i == pan)
                continue;  // don't re-send the potentials list

            cmd.append(sep2);
            cmd.append("LA");
            cmd.append(i);

            Iterator<Integer> pnIter = lan[i].iterator();
            while (pnIter.hasNext())
            {
                cmd.append(sep2);
                int number = pnIter.next().intValue();
                cmd.append(number);
            }
        }

        if (lse != null)
        {
            for (int i = 0; i < lse.length; ++i)
            {
                cmd.append(sep2);
                cmd.append("SE");

                final int[] lse_i = lse[i];
                if ((lse_i.length == 0) && (i == (lse.length - 1)))
                {
                    cmd.append(sep2);
                    cmd.append(0);
                    // 0 is used for padding the last SE list if empty;
                    // otherwise, at the end of the message, an empty list will have no tokens.
                } else {
                    for (int j = 0; j < lse_i.length; ++j)
                    {
                        cmd.append(sep2);
                        int k = lse_i[j];
                        if (k < 0)
                        {
                            cmd.append('-');
                            k = -k;
                        }
                        cmd.append(Integer.toHexString(k));
                    }
                }
            }
        }

        return cmd.toString();

    }

    /**
     * Parse the command String into a PotentialSettlements message
     *
     * @param s   the String to parse
     * @return    a PotentialSettlements message, or null if the data is garbled
     */
    @SuppressWarnings("unchecked")
    public static SOCPotentialSettlements parseDataStr(String s)
    {
        String ga;
        int pn;
        List<Integer> ps = new ArrayList<Integer>();
        HashSet<Integer>[] las = null;
        int pan = 0;
        int[][] legalSeaEdges = null;

        StringTokenizer st = new StringTokenizer(s, sep2);

        try
        {
            ga = st.nextToken();
            pn = Integer.parseInt(st.nextToken());
            boolean hadNA = false;

            while (st.hasMoreTokens())
            {
                String tok = st.nextToken();
                if (tok.equals("NA"))
                {
                    hadNA = true;
                    break;
                }
                ps.add(Integer.valueOf(Integer.parseInt(tok)));
            }

            if (hadNA)
            {
                // If more than 1 land area, the potentialSettlements
                // numbers will be followed by:
                // NA, 3, PAN, 1, LA2, ..., LA3, ...
                // None of the LA#s will be PAN's number.

                final int numArea = Integer.parseInt(st.nextToken());
                las = new HashSet[numArea + 1];

                String tok = st.nextToken();
                if (! tok.equals("PAN"))
                    return null;
                pan = Integer.parseInt(st.nextToken());
                if (pan < 0)
                    return null;

                if (st.hasMoreTokens())
                {
                    tok = st.nextToken();  // "LA2", "LA3", ...
                } else {
                    // should not occur, but allow if 1 area
                    tok = null;
                    if ((numArea > 1) || (pan != 1))
                        return null;
                }

                // las[]: Loop for numAreas, starting with tok == "LA#"
                while (st.hasMoreTokens())
                {
                    if (! tok.startsWith("LA"))
                    {
                        if (tok.equals("SE"))
                            break;  // "SE" are Legal Sea Edges, to be parsed in next loop
                        else
                            return null;  // unrecognized: bad message
                    }

                    final int areaNum = Integer.parseInt(tok.substring(2));
                    HashSet<Integer> ls = new HashSet<Integer>();

                    // Loop for node numbers, until next "LA#"
                    while (st.hasMoreTokens())
                    {
                        tok = st.nextToken();
                        if (tok.equals("SE") || tok.startsWith("LA"))
                            break;
                        ls.add(Integer.valueOf(Integer.parseInt(tok)));
                    }

                    las[areaNum] = ls;
                }

                // legalSeaEdges[][]: Parse the optional "SE" edge lists (SC_PIRI)
                if (st.hasMoreTokens() && tok.equals("SE"))
                {
                    ArrayList<int[]> allLSE = new ArrayList<int[]>();
                    while (st.hasMoreTokens() && tok.equals("SE"))
                    {
                        ArrayList<Integer> lse = new ArrayList<Integer>();

                        // Loop for edge coords, until next "SE"
                        while (st.hasMoreTokens())
                        {
                            tok = st.nextToken();
                            if (tok.equals("SE"))
                                break;
                            final int edge = Integer.parseInt(tok, 16);
                            if (edge != 0)
                                // 0 is used for padding the last SE list if empty;
                                // otherwise, at the end of the message, an empty list will have no tokens.
                                lse.add(Integer.valueOf(edge));
                        }

                        final int L = lse.size();
                        int[] lseArr = new int[L];
                        for (int i = 0; i < L; ++i)
                            lseArr[i] = lse.get(i);

                        allLSE.add(lseArr);
                    }

                    final int L = allLSE.size();
                    legalSeaEdges = new int[L][];
                    for (int i = 0; i < L; ++i)
                        legalSeaEdges[i] = allLSE.get(i);
                }

                // empty ps list is sent as list which is solely {0};
                // if {0} wasn't sent, ps is null
                if (ps.isEmpty())
                    ps = null;
                else if ((ps.size() == 1) && (ps.get(0) == 0))
                    ps.clear();

                if ((las[pan] == null) && (ps != null))
                    las[pan] = new HashSet<Integer>(ps);
                else if (pan != 0)
                    return null;  // not a well-formed message

                // Make sure all LAs are defined
                for (int i = 1; i <= numArea; ++i)
                    if (las[i] == null)
                        return null;
            }
        }
        catch (Exception e)
        {
            return null;
        }

        if (las == null)
            return new SOCPotentialSettlements(ga, pn, ps);
        else
            return new SOCPotentialSettlements(ga, pn, pan, las, legalSeaEdges);
    }

    /**
     * @return a human readable form of the message
     */
    @Override
    public String toString()
    {
        StringBuffer s = new StringBuffer
            ("SOCPotentialSettlements:game=" + game + "|playerNum=" + playerNumber + "|list=");
        if (psNodes.isEmpty())
            s.append("(empty)");
        else if (psNodesFromAll)
            s.append("(fromAllLANodes)");
        else
            for (Integer number : psNodes)
            {
                s.append(Integer.toHexString(number.intValue()));
                s.append(' ');
            }

        if (landAreasLegalNodes != null)
        {
            s.append("|pan=");
            s.append(startingLandArea);
            for (int i = 1; i < landAreasLegalNodes.length; ++i)
            {
                s.append("|la");
                s.append(i);
                s.append('=');
                if (i == startingLandArea)
                {
                    s.append("(psNodes)");
                    continue;
                }

                final HashSet<Integer> nodes = landAreasLegalNodes[i];
                if (nodes.isEmpty())
                {
                    s.append("(empty)");
                    continue;
                }

                Iterator<Integer> laIter = nodes.iterator();
                while (laIter.hasNext())
                {
                    int number = laIter.next().intValue();
                    s.append(Integer.toHexString(number));
                    s.append(' ');
                }
            }
        }

        if (legalSeaEdges != null)
        {
            s.append("|lse={");
            for (int i = 0; i < legalSeaEdges.length; ++i)
            {
                if (i > 0)  s.append(',');
                s.append('{');
                final int[] lse_i = legalSeaEdges[i];
                for (int j = 0; j < lse_i.length; ++j)
                {
                    int k = lse_i[j];
                    if (k < 0)
                    {
                        s.append('-');
                        k = -k;
                    }
                    else if (j > 0)
                    {
                        s.append(',');
                    }
                    s.append(Integer.toHexString(k));
                }
                s.append('}');
            }
            s.append('}');
        }

        return s.toString();
    }

}

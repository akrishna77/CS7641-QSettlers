/**
 * Java Settlers - An online multiplayer version of the game Settlers of Catan
 * Copyright (C) 2003  Robert S. Thomas <thomas@infolab.northwestern.edu>
 * Portions of this file Copyright (C) 2009-2012,2014,2016-2017 Jeremy D Monin <jeremy@nand.net>
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

import soc.server.genericServer.Connection;

import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;


/**
 * This message lists all the members of a game: Players and observers.
 *<P>
 * In response to {@link SOCJoinGame JOINGAME}, the joining player is sent a specific
 * sequence of messages with details about the game: Board layout, player scores,
 * piece counts, etc. This sequence ends with GAMEMEMBERS, SETTURN and GAMESTATE.
 * GAMEMEMBERS thus tells the client that the server is ready for its input.
 *<P>
 * Robots use GAMEMEMBERS as their cue to sit down at the game, if they've been
 * asked to sit from {@link SOCBotJoinGameRequest BOTJOINGAMEREQUEST}.
 * In order for the robot to be certain it has all details about a game,
 * bots should take no action before receiving GAMEMEMBERS.
 *<P>
 * When forming a new game, clients will be sent the sequence as described above, and
 * then will each choose a position and sit down. Any client can then send {@link SOCStartGame}
 * to start the game. Server responds with the newly generated board layout and
 * other game and board details, any scenario-specific {@link SOCPutPiece}s, then
 * {@link SOCGameState}({@link soc.game.SOCGame#START1A START1A}), then finally {@link SOCStartGame}.
 *
 * @author Robert S Thomas
 * @see SOCChannelMembers
 * @see SOCGamesWithOptions
 */
public class SOCGameMembers extends SOCMessage
    implements SOCMessageForGame
{
    private static final long serialVersionUID = 2000L;  // last structural change v2.0.00

    /**
     * List of members
     */
    private List<String> members;

    /**
     * Name of game
     */
    private String game;

    /**
     * Create a GameMembers message.
     *
     * @param ga  name of game
     * @param ml  list of members
     */
    public SOCGameMembers(String ga, List<String> ml)
    {
        messageType = GAMEMEMBERS;
        members = ml;
        game = ga;
    }

    /**
     * @return the list of member names; each element is a String with the member's nickname
     */
    public List<String> getMembers()
    {
        return members;
    }

    /**
     * @return the game name
     */
    public String getGame()
    {
        return game;
    }

    /**
     * GAMEMEMBERS sep game sep2 members
     *
     * @return the command String
     */
    @Override
    public String toCmd()
    {
        return toCmd(game, members);
    }

    /**
     * GAMEMEMBERS sep game sep2 members
     *<P>
     * Used from instance method {@link #toCmd()} with Strings,
     * and from other callers with {@link Connection}s for convenience.
     *
     * @param ga  the game name
     * @param ml  the list of members (String or {@link Connection})
     * @return    the command string
     */
    public static String toCmd(String ga, List<?> ml)
    {
        String cmd = GAMEMEMBERS + sep + ga;

        try
        {
            for (Object obj : ml)
            {
                String str;
                if (obj instanceof Connection)
                {
                    str = ((Connection) obj).getData();
                }
                else if (obj instanceof String)
                {
                    str = (String) obj;
                }
                else
                {
                    str = obj.toString();  // fallback; expecting String or conn
                }

                cmd += (sep2 + str);
            }
        }
        catch (Exception e) {}

        return cmd;
    }

    /**
     * Parse the command String into a GAMEMEMBERS message.
     *
     * @param s   the String to parse
     * @return    a GAMEMEMBERS message, or null if the data is garbled
     */
    public static SOCGameMembers parseDataStr(String s)
    {
        String ga;
        List<String> ml = new ArrayList<String>();
        StringTokenizer st = new StringTokenizer(s, sep2);

        try
        {
            ga = st.nextToken();

            while (st.hasMoreTokens())
                ml.add(st.nextToken());
        }
        catch (Exception e)
        {
            return null;
        }

        return new SOCGameMembers(ga, ml);
    }

    /**
     * @return a human readable form of the message
     */
    @Override
    public String toString()
    {
        StringBuffer sb = new StringBuffer("SOCGameMembers:game=");
        sb.append(game);
        sb.append("|members=");
        if (members != null)
            sb.append(members);  // "[joe, bob, lily,...]"
        return sb.toString();
    }

}
